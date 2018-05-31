#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Implementation of KNN

# Author      :  Vitor Rodrigues Di Toro
# E-Mail      :  vitorrditoro@gmail.com
# Create on   :  19/03/2018
# Last Update :  30/05/2018

from sources.distances import DistanceType, Distance
from sources.dataSetUtils import DataSet


class KNN:
    """

    refs: [1] - https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
          [2] - http://madhugnadig.com/articles/machine-learning/2017/01/13/implementing-k-nearest-neighbours-from-scratch-in-python.html

    """
    def __init__(self, training, test):
        self.training = training
        self.test = test
        self.training_size = len(training)
        self.test_size = len(test)
        self.accuracy = None
        self.recall = None
        self.precision = None
        self.f1_score = None
        self._tp = None
        self._fp = None
        self._tn = None
        self._fn = None

    def _prepare_metrics(self, result):
        self._tp = 0.0
        self._fp = 0.0
        self._tn = 0.0
        self._fn = 0.0

        for i in range(self.test_size):

            # Count true positives
            if result[i] == 'g' and self.test[i][-1] == 'g':
                self._tp += 1.0

            # Count false positives
            if result[i] == 'g' and self.test[i][-1] == 'b':
                self._fp += 1.0

            # Count true negatives
            if result[i] == 'b' and self.test[i][-1] == 'b':
                self._tn += 1.0

            # Count false negatives
            if result[i] == 'b' and self.test[i][-1] == 'g':
                self._fn += 1.0

    def _calc_accuracy(self):
        # TODO --> Fix DocString
        """

        ref: http://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/
        """
        # Accuracy = (TP + TN) / (TP + FP + TN + FN)
        self.accuracy = (self._tp + self._tn) / (self._tp + self._fp + self._tn + self._fn)

    def _calc_precision(self):
        # TODO --> Fix DocString
        """

        ref: http://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/
        """
        # Precision = TP / (TP + FP)
        self.precision = self._tp / (self._tp + self._fp)

    def _calc_recall(self):
        # TODO --> Fix DocString
        """

        ref: http://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/
        """
        # Recall = TP / (TP + FN)
        self.recall = self._tp / (self._tp + self._fn)

    def _calc_f1_score(self):
        # TODO --> Fix DocString
        """

        ref: http://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/
        """
        # F1 Score = 2.0 * (Recall * Precision) / (Recall + Precision)
        self.f1_score = 2.0 * (self.recall * self.precision) / (self.recall + self.precision)

    def fit(self, k: int, distance_method: DistanceType, distance_order=0.5):
        """
        :param k: number of neighbors
        :param distance_method: method to calculate distance (Euclidean, Manhattan or Minkowski)
        :param distance_order: distance order (Just used in Minkowski distance)
        """
        result = []

        for i in range(self.test_size):
            distances = {}

            distance = Distance()
            distance.set_distance_order(distance_order)

            for j in range(self.training_size):
                distances[j] = distance.calculator(self.test[i], self.training[j], distance_method)

            k_neighbors = sorted(distances, key=distances.get)[:k]

            #  TODO:
            #  -  Alterar a votação majoritária para lidar com os indices de um dicionário
            #     e não valores hardcoded pré estabelecidos.

            g_count, b_count = 0, 0

            for index in k_neighbors:
                if self.training[index][-1] == 'g':
                    g_count += 1
                else:
                    b_count += 1

            if g_count > b_count:
                result.append('g')
            else:
                result.append('b')

        self._prepare_metrics(result)
        self._calc_accuracy()
        self._calc_precision()
        self._calc_recall()
        self._calc_f1_score()


def main():
    data_set_name = '../dataset/ionosphere.csv'
    training_data, test_data = DataSet.get_data(data_set_name, percent_to_training=60, randomize=True, verbose=False)

    knn = KNN(training_data, test_data)

    k = 13
    print("\nEuclidean distance:")
    knn.fit(k=k, distance_method=DistanceType.EUCLIDEAN)
    print("Accuracy: %.4f" % knn.accuracy)

    print("\nManhattan distance:")
    knn.fit(k=k, distance_method=DistanceType.MANHATTAN)
    print("Accuracy: %.4f" % knn.accuracy)

    print("\nMinkowski distance:")
    knn.fit(k=k, distance_method=DistanceType.MINKOWSKI, distance_order=0.5)
    print("Accuracy: %.4f" % knn.accuracy)


if __name__ == '__main__':
    main()
