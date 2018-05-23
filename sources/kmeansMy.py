#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Implementação do KNN

# Author      :  Vitor Rodrigues Di Toro
# E-Mail      :  vitorrditoro@gmail.com
# Date        :  09/05/2018
# Last Update :  22/05/2018

import sys

import random
import statistics
from distances import Distance
#from tsne import tsne 
#import numpy
#import pylab

import csv
from random import shuffle

sys.path.append('../')


def get_data_lc(dataset_name, lines, columns, randomize=False):
    with open(dataset_name, 'r') as File:
        reader = csv.reader(File, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)

        l_reader = list(reader)

        if randomize:
            shuffle(l_reader)

    data = []
    tmp = [[l_reader[i][j] for j in columns] for i in lines]
    for row in tmp:
        data.append([num(i) for i in row])

    return data


def num(s):
    try:
        return float(s)
    except ValueError:
        return s


class Clusters(dict):
    """
    Referências do K-Means:

    [1] Artigo de autoria de Madhu G. Nadig (https://github.com/madhug-nadig)
        Disponível em: https://bit.ly/2GJTLO6

    [2] Artigo de autoria de Matthew Mayo (https://github.com/mmmayo13)
        Disponível em: https://bit.ly/2pRWH0Z

    [3] Artigo de autoria de Mubaris NK (https://github.com/mubaris)
        Disponível em:  https://bit.ly/2sAS4Ng
    """

    def __init__(self):
        pass

    def iter_columns(self, cluster):
        try:
            return range(len(self[cluster][0]))
        except Exception:
            return []

    def iter_lines(self, cluster):
        try:
            return range(len(self[cluster]))
        except Exception:
            return []


class Kmeans:

    def __init__(self, k=2, tolerance=0.0001, max_iterations=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.centroids = []
        self.clusters = Clusters()
        self.labels = []
        self.iteration = 0

    def initialize_cluster(self):
        """
        Inicializa a K clusters, como listas vazias.

        :return:
        """

        self.clusters = Clusters()
        for i in range(self.k):
            self.clusters[i] = []

    def initialize_centoids(self, data):
        """
        Inicializa as K centroides, em posições aleatórias dentro do limite de Max e Min.

        :param data:
        :return:
        """

        self.centroids = []
        columns_max = []
        columns_min = []

        # Cálcula o Max e o Min de cada coordenada (coluna)
        for j in range(len(data[0])):
            temp = [data[i][j] for i in range(len(data))]
            columns_max.append(max(temp))
            columns_min.append(min(temp))

        # Cria K ponto (centroid) aleatórios, entre o valor máximo e mínimo de cada coordenada
        for i in range(self.k):
            p = []
            for i_max, i_min in zip(columns_max, columns_min):
                p.append(random.uniform(i_min, i_max))

            self.centroids.append(p)

        # print("\nCentroides inicializados:")
        # print(self.centroids)
        # print("\n")

    def update_centroids(self):
        """
         Atualiza as K centroides, deslocando-as para o ponto médio de seus respctivos cluester.
        :return:
        """

        for cluster in self.clusters:
            p = []
            for j in self.clusters.iter_columns(cluster):
                column = [self.clusters[cluster][i][j] for i in self.clusters.iter_lines(cluster)]
                p.append(statistics.mean(column))

            self.centroids[cluster] = p

    def classifies_points(self, data, distance_method, distance_order=0.5):
        """
        :param data:
        :param distance_method:
        :param distance_order:
        :return:

        Calcula a distancia de todos os pontos para cada centroide
        Classifica cada ponto do conjunto data como pertecendo a um dos clusters (centroids)
        """

        distance_calculator = Distance.Calculator

        for row in data:
            if distance_method == Distance.Type.euclidean:
                distances = [distance_calculator.euclidean_distance(row, centroid) for centroid in self.centroids]
            elif distance_method == Distance.Type.manhattan:
                distances = [distance_calculator.manhattan_distance(row, centroid) for centroid in self.centroids]
            elif distance_method == Distance.Type.minkowski:
                distances = [distance_calculator.minkowski_distance(row, centroid, distance_order) for centroid in self.centroids]

            cluster_type = distances.index(min(distances))
            self.clusters[cluster_type].append(row)

    def stop_threshold(self, list_a, list_b, distance_method, distance_order=0.5):
        """

        :param list_a:
        :param list_b:
        :param distance_method:
        :param distance_order:
        :return:
        """

        distance_calculator = Distance.Calculator

        for a, b in zip(list_a, list_b):
            if distance_method == Distance.Type.euclidean:
                if distance_calculator.euclidean_distance(a, b) > self.tolerance:
                    return False
            elif distance_method == Distance.Type.manhattan:
                if distance_calculator.manhattan_distance(a, b) > self.tolerance:
                    return False
            elif distance_method == Distance.Type.minkowski:
                if distance_calculator.minkowski_distance(a, b, distance_order) > self.tolerance:
                    return False

        return True

    def fit(self, data, distance_method=Distance.Type.euclidean, distance_order=0.5):

        changed = True
        iteration = 0

        # Inicializa as K centroides (posições aleatórias)
        self.initialize_centoids(data)

        while changed:  # ....repete a porra toda

            # Update iteration number
            iteration += 1

            # Inicializa a K clusters, como uma listas vazias.
            self.initialize_cluster()

            # Calcula a distancia de todos os pontos para cada centroide
            # Classifica cada ponto do conjunto data como pertecendo a um dos clusters (centroids)
            self.classifies_points(data, distance_method, distance_order)

            # Salva a posição atual das centroids
            previous = self.centroids.copy()

            # Atualiza as K centroides, deslocando cada centroide para o ponto médio de seu cluester
            self.update_centroids()

            # print("Centroides %d" %iteration)
            # self.print_centroids()

            # Verifica critério de parada
            if self.max_iterations <= iteration:
                self.initialize_cluster()
                self.classifies_points(data, distance_method)
                changed = False
                # print("Sai em 1.")
            elif previous == self.centroids:
                changed = False

            # TODO: Fix it
            # elif self.stop_threshold(previous, self.centroids,distanceMethod):
            #    self.initialize_cluster()
            #    self.classifies_points(data, distanceMethod)
            #    changed = False
            #    print("Sai em 3.")

        self.iteration = iteration

    def print_centroids(self):
        for centroid in self.centroids:
            print(centroid)
        print("\n")
        
    def labels_(self,data):
        labels = []
        for i in data:
            for j in range(0,len(self.clusters)):
                if i in self.clusters[j]:
                    labels.append(j)
        return labels


def main():
    data = get_data_lc('../dataset/ionosphere.csv', range(350), (0, 1), randomize=True)

    kms = Kmeans(k=2, max_iterations=500)
    kms.fit(data, distance_method=Distance.Type.euclidean)

    print("Pontos do tipo 1: %d" % len(kms.clusters[0]))
    print("Pontos do tipo 2: %d" % len(kms.clusters[1]))
    #Y = tsne.tsne(numpy.asarray(kms.clusters[0]),initial_dims=34)
    #pylab.scatter(Y[:, 0], Y[:, 1], 20)
    #pylab.show()
    #Y = tsne.tsne(numpy.asarray(kms.clusters[1]),initial_dims=34)
    #pylab.scatter(Y[:, 0], Y[:, 1])
    #pylab.show()
    #print("Pontos do tipo 3: %d" % len(kms.clusters[2]))

    #from matplotlib import pyplot as plt

    #for x, y in kms.clusters[0]:
       # plt.scatter(x, y, c='red', s=7)

    #for x, y in kms.clusters[1]:
       # plt.scatter(x, y, c='blue', s=7)

   # for x, y in kms.clusters[2]:
    #    plt.scatter(x, y, c='green', s=7)

   # plt.show()
   

if __name__ == '__main__':
    main()
