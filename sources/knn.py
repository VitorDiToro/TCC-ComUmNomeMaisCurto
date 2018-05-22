#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Implementação do KNN 

# Author      :  Vitor Rodrigues Di Toro
# E-Mail      :  vitorrditoro@gmail.com
# Date        :  19/03/2018
# Last Update :  19/05/2018

import sys
sys.path.append('../')

from sources.distances import *
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

    def __calc_accuracy__(self, result):
        # TODO --> Fix DocString
        """

        :param result:
        :return:
        """

        score = 0
        for i in range(self.test_size):

            if result[i] == self.test[i][-1]:
                score += 1

        self.accuracy = (100 * score / self.test_size)

    def fit(self, k: int, distance_method: Distance.Type, distance_order=0.5):
        """

        :param k:
        :param distance_method:
        :param distance_order:
        :return:
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

        self.__calc_accuracy__(result)


def main():
    data_set_name = '../dataset/ionosphere.csv'
    training_data, test_data = DataSet.get_data(data_set_name, percent_to_training=60, randomize=True, verbose=False)

    knn = KNN(training_data, test_data)

    k = 13
    print("\nEuclidean distance:")
    knn.fit(k=k, distance_method=Distance.Type.euclidean())
    print("Accuracy: %.4f %%" % knn.accuracy)

    print("\nManhattan distance:")
    knn.fit(k=k, distance_method=Distance.Type.manhattan())
    print("Accuracy: %.4f %%" % knn.accuracy)

    print("\nMinkowski distance:")
    knn.fit(k=k, distance_method=Distance.Type.minkowski(), distance_order=0.5)
    print("Accuracy: %.4f %%" % knn.accuracy)


if __name__ == '__main__':
    main()
