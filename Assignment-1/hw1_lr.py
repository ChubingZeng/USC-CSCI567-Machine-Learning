from __future__ import division, print_function

from typing import List

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        trans_features = numpy.append(numpy.ones((len(features),1)),features,axis=1)
        self.weight = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(numpy.transpose(trans_features),trans_features)),
                                    numpy.transpose(trans_features)),values).tolist()

    def predict(self, features: List[List[float]]) -> List[float]:
        trans_features = numpy.append(numpy.ones((len(features),1)),features,axis=1)
        predict = numpy.dot(trans_features,self.weight)
        return predict

    def get_weights(self) -> List[float]:
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.weight


class LinearRegressionWithL2Loss:
    '''Use L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        trans_features = numpy.append(numpy.ones((len(features),1)),features,axis=1)
        self.weight = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(numpy.transpose(trans_features),trans_features)
                                                  +self.alpha*numpy.identity(numpy.array(trans_features).shape[1])),
                                    numpy.transpose(trans_features)),values).tolist()


    def predict(self, features: List[List[float]]) -> List[float]:
        trans_features = numpy.append(numpy.ones((len(features),1)),features,axis=1)
        predict = numpy.dot(trans_features,self.weight)
        return predict

    def get_weights(self) -> List[float]:
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        return self.weight


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
