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
    def __init__(self, training, test):
        self.training = training
        self.test = test
        self.training_size = len(training)
        self.test_size = len(test)
        self.accuracy = -1
        self.recall = -1
        self.precision = -1
        self.f1_score = -1
        self._number_of_positives = -1
        self._positive_hits = -1
        self._classified_as_positive = -1

    def _prepare_metrics(self, result):
        self.hits = 0
        self._number_of_positives = 0.0
        self._positive_hits = 0.0
        self._classified_as_positive = 0.0

        for i in range(self.test_size):
            # count hits
            if result[i] == self.test[i][-1]:
                self.hits += 1.0                                # Count TP and TN

            # count positives classifications
            if result[i] == 'g':
                self._classified_as_positive += 1.0             # Count TP and FP

            # count positive hits
            if result[i] == 'g' and self.test[i][-1] == 'g':
                self._positive_hits += 1.0                      # Count TP

            # count number of positives values in Test Group
            if self.test[i][-1] == 'g':
                self._number_of_positives += 1.0                # Count TP and FN

    def _calc_accuracy(self):
        # TODO --> Fix DocString
        """

        ref: http://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/
        """
        # Accuracy = (TP + TN) / (TP + FP + TN + FN)
        self.accuracy = self.hits / self.test_size

    def _calc_precision(self):
        # TODO --> Fix DocString
        """

        ref: http://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/
        """
        # Precision = TP / (TP + FN)
        self.precision = self._positive_hits / self._classified_as_positive

    def _calc_recall(self):     # OK, @Vitor, 30/05/18
        # TODO --> Fix DocString
        """

        ref: http://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/
        """
        # Recall = TP / (TP + FN)
        self.recall = self._positive_hits / self._number_of_positives

    def _calc_f1_score(self):   # OK, @Vitor, 30/05/18
        # TODO --> Fix DocString
        """

        ref: http://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/
        """
        # F1 Score = 2*(Recall * Precision) / (Recall + Precision)
        self.f1_score = 2 * (self.recall * self.precision)/(self.recall + self.precision)

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
