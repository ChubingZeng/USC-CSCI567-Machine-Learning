from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        '''
            Args : 
            nb_features : Number of features
            max_iteration : maximum iterations. You algorithm should terminate after this
            many iterations even if it is not converged 
            margin is the min value, we use this instead of comparing with 0 in the algorithm
        '''
        
        self.nb_features=nb_features
        self.w = [0 for i in range(0,nb_features+1)]
        self.margin = margin
        self.max_iteration = max_iteration
    
    def sign_vec(self,x):
        convert = [0 for i in range(len(x))]
        for i in range(len(convert)):
            convert[i] = 1 if x[i]>=0 else -1
        return convert

    def sign(self,x):
        if x>=0:
            return 1
        if x<0: 
            return -1

    def norm(self,x):
        return np.sqrt(np.inner(x,x))

    
    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            labels : label of each feature [-1,1]
            
            Returns : 
                True/ False : return True if the algorithm converges else False. 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and should update 
        # to correct weights w. Note that w[0] is the bias term. and first term is 
        # expected to be 1 --- accounting for the bias
        ############################################################################
        iteration = 0 
        converge = False
        weight = self.w
        max_iteration = self.max_iteration
        margin = self.margin
        sign = self.sign
        sign_vec=self.sign_vec
        norm = self.norm
        
        while iteration <=  max_iteration and not converge:
            for i in range(len(labels)):
                pred = sign(np.dot(weight,np.transpose(features[i])))
                if pred != labels[i]:
                    weight = weight + labels[i]*np.array(features[i])/norm(features[i])
            label_pred = sign_vec(np.dot(weight,np.transpose(features)))
            iteration += 1
            if norm(np.subtract(label_pred,labels)) < margin:
                converge = True
        self.w = weight
        self.converge = converge
        
        return self.converge

        
    
    def reset(self):
        self.w = [0 for i in range(0,self.nb_features+1)]
        
    def predict(self, features: List[List[float]]) -> List[int]:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            
            Returns : 
                labels : List of integers of [-1,1] 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and use the learned 
        # weights to predict the label
        ############################################################################
        sign_vec = self.sign_vec
        prediction = sign_vec(np.dot(self.w,np.transpose(features)))
        return prediction

    def get_weights(self) -> List[float]:
        return self.w
    