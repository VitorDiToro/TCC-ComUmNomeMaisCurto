#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Implementação do KNN 

# Author      :  Vitor Rodrigues Di Toro
# E-Mail      :  vitorrditoro@gmail.com
# Date        :  19/03/2018
# Last Update :  07/05/2018

import csv
from sources.distances import Distance
from random import shuffle

import sys
sys.path.append('../')


def num(s):
    try:
        return float(s)
    except ValueError:
        return s


def get_data(dataset_name: str = 'ionosphere.csv', percent_to_training=60, randomize=True, verbose=True):
    count = 0
    group_g = 0
    test_data = []
    training_data = []

    with open(dataset_name, 'r') as File:
        reader = csv.reader(File, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)

        l_reader = list(reader)
        limit = int(len(l_reader) * (percent_to_training / 100))

        if randomize:
            shuffle(l_reader)

        for row in l_reader:

            if row[-1] == "g":
                group_g += 1

            if count < limit:
                training_data.append([num(i) for i in row])
            else:
                test_data.append([num(i) for i in row])

            count += 1

    if verbose:
        print("Total de amostras: %d:" % len(l_reader))
        print("    - %d amostra do tipo \"Good\"" % group_g)
        print("    - %d amostra do tipo \"Bad\"" % (len(l_reader) - group_g))
        print("")
        print("%d %% das amostras separadas para treino." % percent_to_training)
        print("    - %d amostras para treino" % (len(training_data)))
        print("    - %d amostras para teste" % (len(test_data)))

    return training_data, test_data


class KNN:
    def __init__(self, training, test):
        self.training = training
        self.test = test
        self.training_size = len(training)
        self.test_size = len(test)
        self.accuracy = -1

    def calc_accuracy(self, result):
        score = 0
        for i in range(self.test_size):

            if result[i] == self.test[i][-1]:
                score += 1

        self.accuracy = (100 * score / self.test_size)

    def fit(self, k: int, distance_method: int, distance_order=0.5):

        result = []

        for i in range(self.test_size):
            distances = {}

            distance_calculator = Distance.Calculator

            for j in range(self.training_size):
                if distance_method == Distance.Type.euclidean:
                    distances[j] = distance_calculator.euclidean_distance(self.test[i], self.training[j])
                elif distance_method == Distance.Type.manhattan:
                    distances[j] = distance_calculator.manhattan_distance(self.test[i], self.training[j])
                elif distance_method == Distance.Type.minkowski:
                    distances[j] = distance_calculator.minkowski_distance(self.test[i], self.training[j],
                                                                          distance_order)

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

        self.calc_accuracy(result)


def main():
    training_data, test_data = get_data('../dataset/ionosphere.csv', 60, randomize=True, verbose=False)

    knn = KNN(training_data, test_data)

    print("\nEuclidean distance:")
    knn.fit(k=13, distance_method=Distance.Type.euclidean)
    print("Accuracy: %.4f %%" % knn.accuracy)

    print("\nManhattan distance:")
    knn.fit(k=13, distance_method=Distance.Type.manhattan)
    print("Accuracy: %.4f %%" % knn.accuracy)

    print("\nMinkowski distance:")
    knn.fit(k=13, distance_method=Distance.Type.minkowski, distance_order=0.5)
    print("Accuracy: %.4f %%" % knn.accuracy)


if __name__ == '__main__':
    main()
