import numpy as np
from typing import List
from classifier import Classifier

class DecisionStump(Classifier):
    def __init__(self, s:int, b:float, d:float):
        self.clf_name = "Decision_stump"
        self.s = s
        self.b = b
        self.d = d
    
    def train(self, features: List[List[float]], labels: List[int]):
        pass
    
    def predict(self, features: List[List[float]]) -> List[int]:
        features = np.array(features)
        N,D = features.shape
        preds = np.zeros(N) 
        preds[features[:,self.d] > self.b] = self.s
        preds[features[:,self.d] <= self.b] = -self.s
        return preds.astype(int)
    
		