from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:
    
    def __init__(self, k: int, distance_function) -> float:
        self.k = k
        self.distance_function = distance_function

    def train(self, features: List[List[float]], labels: List[int]):
        self.train_features = features
        self.train_labels = labels

    def predict(self, features_new: List[List[float]]) -> List[int]:
        features_train = self.train_features
        labels_train = self.train_labels
        k = self.k
        dist = numpy.full((numpy.array(features_new).shape[0],numpy.array(features_train).shape[0]), numpy.nan)
        for i in range(dist.shape[0]):
            for j in range(dist.shape[1]):
                dist[i,j] = self.distance_function(features_new[i],features_train[j])
        neighbors = dist.argsort()[:,:k]
        vote = numpy.full((numpy.array(features_new).shape[0],k), numpy.nan)
        for l in range(vote.shape[0]):
            for m in range(vote.shape[1]):
                vote[l,m] = labels_train[neighbors[l,m]]
        vote=vote.astype(int)
        predicted = scipy.stats.mode(vote,axis=1)[0][:,0].tolist()
        
        return predicted


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
