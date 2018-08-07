from typing import List

import numpy as np


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    return np.mean(np.subtract(y_true,y_pred)**2)



def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)
    try: 
        addup = [x + y for x, y in zip(real_labels, predicted_labels)]
        tp = np.sum(np.equal(addup,2)) # true positive
        precision = sum(np.equal(addup,2))/sum(np.equal(predicted_labels,1))
        recall = sum(np.equal(addup,2))/sum(np.equal(real_labels,1))
        if precision == 0 and recall ==0:
            f1_score = 0
        else: 
            f1_score = 2*(precision*recall)/(precision+recall)
        #f1_score = np.nan_to_num(f1_score)
    except: 
        f1_score = 0
    return f1_score


def polynomial_features(features: List[List[float]], k: int) -> List[List[float]]:
    p = 2
    concate = features
    while p <= k:
        temp = np.power(features,p)
        concate = np.concatenate((concate,temp),axis = 1)
        p = p+1 
    return concate


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    sub = np.subtract(point1,point2)
    return np.sqrt(np.inner(sub,sub))


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    return np.inner(point1,point2)


def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    sub = np.subtract(point1,point2)
    return -np.exp(-1/2*np.inner(sub,sub))


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        inner = np.full(np.array(features).shape[0], np.nan)
        for j in range(np.array(features).shape[0]):
            inner[j] = np.sqrt(np.inner(np.array(features)[j],np.array(features)[j]))
        
        features_prime = np.full((np.array(features).shape[0],np.array(features).shape[1]), np.nan)
        for j in range(np.array(features).shape[0]):
            features_prime[j] = np.array(features)[j]/inner[j]
        features_prime = np.nan_to_num(features_prime)
        
        return features_prime


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        pass
    
    count = 0
    
    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        self.count += 1
        if self.count <= 1:
            self.feature_fix = features
            self.gmin = np.amin(features, axis=0) 
            self.gmax = np.amax(features, axis=0) 
            self.gdiff = np.subtract(self.gmax,self.gmin)
    
        features_scale = np.full((np.array(features).shape[0],np.array(features).shape[1]), np.nan)
        for i in range(np.array(features).shape[0]):
            for j in range(np.array(features).shape[1]):
                features_scale[i,j] = (np.array(features)[i,j] - self.gmin[j])/self.gdiff[j]
        
        return features_scale  

