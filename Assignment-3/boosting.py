import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
    # Boosting from pre-defined classifiers
    def __init__(self, clfs: Set[Classifier], T=0):
        self.clfs = clfs
        um_clf = len(clfs)
        if T < 1:
            self.T = um_clf
        else:
            self.T = T
        
        self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
        self.betas = []       # list of weights beta_t for t=0,...,T-1
        return
    
    @abstractmethod
    def train(self, features: List[List[float]], labels: List[int]):
        return
    
    def predict(self, features: List[List[float]]) -> List[int]:
        ########################################################
        # TODO: implement "predict"
        ########################################################
        features = np.array(features)
        N,D = features.shape
        pred = np.zeros((N,self.T))
        for i in range(self.T):
            pred[:,i] = self.betas[i] * self.clfs_picked[i].predict(features)
        predicted = np.sign(np.sum(pred,axis = 1)).astype(int).tolist()
        return predicted


class AdaBoost(Boosting):
    def __init__(self, clfs: Set[Classifier], T=0):
        Boosting.__init__(self, clfs, T)
        self.clf_name = "AdaBoost"
        return
    
    def train(self, features: List[List[float]], labels: List[int]):
        ############################################################
        # TODO: implement "train"
        ############################################################
        features = np.array(features)
        N,D = features.shape
        w = np.full((1,N),1/N)
        h = list(self.clfs)
        for i in range(self.T):
            tochoose = []
            for hfun in self.clfs:
                tochoose.append(np.sum(w * np.where(hfun.predict(features) == labels,0,1)))   
            h_t = h[np.argmin(tochoose)]
            self.clfs_picked.append(h_t)
            e_t = np.min(tochoose)
            beta_t = 1/2*np.log((1-e_t)/e_t)
            self.betas.append(beta_t)
            w_t1 = np.where(h_t.predict(features) == labels,w*np.exp(-beta_t),w*np.exp(beta_t))
            w = w_t1/np.sum(w_t1)
        return 
    
    def predict(self, features: List[List[float]]):
        
        return Boosting.predict(self, features)



class LogitBoost(Boosting):
    def __init__(self, clfs: Set[Classifier], T=0):
        Boosting.__init__(self, clfs, T)
        self.clf_name = "LogitBoost"
        return
    
    def train(self, features: List[List[float]], labels: List[int]):
        ############################################################
        # TODO: implement "train"
        ############################################################
        features = np.array(features)
        labels = np.array(labels)
        N,D = features.shape
        my_pi = np.full((1,N),1/2)
        h = list(self.clfs)
        f = np.zeros(len(labels))
        
        for i in range(self.T):
            z_t = ((labels + 1)/2 - my_pi)/(my_pi*(1-my_pi))
            w_t = my_pi*(1-my_pi)
            tochoose = []
            for hfun in self.clfs:
                tochoose.append(np.sum(w_t *np.square((z_t - hfun.predict(features)))))   
            h_t = h[np.argmin(tochoose)]
            self.clfs_picked.append(h_t)
            f = f + 1/2*h_t.predict(features)
            my_pi = 1/(1+np.exp(-2*f))
            self.betas.append(1/2)
        return
    
    def predict(self, features: List[List[float]]) -> List[int]:
        return Boosting.predict(self, features)
	