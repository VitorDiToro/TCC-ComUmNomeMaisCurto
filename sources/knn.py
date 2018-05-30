#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Implementação do KNN 

# Author      :  Vitor Rodrigues Di Toro
# E-Mail      :  vitorrditoro@gmail.com
# Create      :  19/03/2018
# Last Update :  24/05/2018

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
        self.__number_of_positives__ = -1
        self.__positive_hits__ = -1
        self.__classified_as_positive__ = -1

    def _calc_accuracy(self, result):
        # TODO --> Fix DocString
        """

        :param result:
        :return:
        """

        score = 0
        self.__number_of_positives__ = 0.0
        self.__positive_hits__ = 0.0
        self.__classified_as_positive__ = 0.0

        for i in range(self.test_size):
            # count hits
            if result[i] == self.test[i][-1]:
                score += 1

            # count positives classifications
            if result[i] == 'g':
                self.__classified_as_positive__ = self.__classified_as_positive__ + 1.0

            # count positive hits
            if result[i] == 'g' and self.test[i][-1] == 'g':
                self.__positive_hits__ = self.__positive_hits__ + 1.0

            # count number of positives values in Test Group
            if self.test[i][-1] == 'g':
                self.__number_of_positives__ = self.__number_of_positives__ + 1.0

        self.accuracy = score / self.test_size

    def _precision(self):
        # TODO --> Fix DocString
        """

        """
        self.precision = self.__positive_hits__ / self.__number_of_positives__

    def _calc_recall(self):
        # TODO --> Fix DocString
        """

        """
        self.recall = self.__positive_hits__ / self.__classified_as_positive__

    def _calc_f1_score(self):
        # TODO --> Fix DocString
        """

        """
        self.f1_score = 2 * (self.precision * self.recall)/(self.precision + self.recall)

    def fit(self, k: int, distance_method: DistanceType, distance_order=0.5):
        # TODO --> Fix DocString
        """
        :param k:
        :param distance_method:
        :param distance_order:
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

        self._calc_accuracy(result)
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
