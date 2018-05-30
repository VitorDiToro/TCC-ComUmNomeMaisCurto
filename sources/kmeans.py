#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Implementação do K-Means

# Authors     :  Vitor Rodrigues Di Toro <vitorrditoro@gmail.com>
#                Jonatan Alberto Afonso  <joalberto1@hotmail.com>
# Create  on  :  09/05/2018
# Last Update :  30/05/2018


import random
import statistics

from sources.dataSetUtils import DataSet
from sources.distances import Distance, DistanceType


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
        super().__init__()
        pass

    def iter_columns(self, cluster):
        try:
            return range(len(self[cluster][0]))
        except ValueError:
            return []

    def iter_lines(self, cluster):
        try:
            return range(len(self[cluster]))
        except ValueError:
            return []


class KMeans:

    def __init__(self, k=2, tolerance=0.0001, max_iterations=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.centroids = []
        self.clusters = Clusters()
        self.labels = []
        self.iteration = 0

    def _recover_labels(self, data):
        labels = []
        for i in data:
            for j in range(0, len(self.clusters)):
                if i in self.clusters[j]:
                    labels.append(j)
        self.labels = labels

    def initialize_cluster(self):
        """
        Inicializa a K clusters, como listas vazias.

        :return:
        """

        self.clusters = Clusters()
        for i in range(self.k):
            self.clusters[i] = []

    def initialize_centroids(self, data):
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

        distance = Distance()
        distance.set_distance_order(distance_order)

        for row in data:

            distances = [distance.calculator(row, centroid, distance_method) for centroid in self.centroids]

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

        distance = Distance()
        distance.set_distance_order(distance_order)

        for a, b in zip(list_a, list_b):
            if distance.calculator(a, b, distance_method) > self.tolerance:
                return False

        return True

    def fit(self, data, distance_method=DistanceType.EUCLIDEAN, distance_order=0.5):

        changed = True
        iteration = 0

        # Inicializa as K centroides (posições aleatórias)
        self.initialize_centroids(data)

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
                # print("=== Número máximo de iterações atingido! === \n")
                changed = False
            elif previous == self.centroids:
                # print("=== Sistema convergiu! \o/ === \n")
                changed = False
            elif self.stop_threshold(previous, self.centroids, distance_method):
                self.initialize_cluster()
                self.classifies_points(data, distance_method)
                # print('=== Threshold atingido=== \n')
                changed = False

        self.iteration = iteration
        # print("Iteration: %d" % iteration)

        self._recover_labels(data)

    def print_centroids(self):
        for centroid in self.centroids:
            print(centroid)
        print("\n")


def main():
    data = DataSet.get_data_lc('../dataset/ionosphere.csv', range(350), range(34), randomize=True)

    kms = KMeans(k=2, max_iterations=500)
    kms.fit(data, distance_method=DistanceType.EUCLIDEAN)

    print("Nº de iterações : " + str(kms.iteration))
    print("Pontos do tipo 1: " + str(len(kms.clusters[0])))
    print("Pontos do tipo 2: " + str(len(kms.clusters[1])))
    """
    print("Pontos do tipo 3: %d" % len(kms.clusters[2]))

    from matplotlib import pyplot as plt

    for x, y in kms.clusters[0]:
        plt.scatter(x, y, c='red', s=7)

    for x, y in kms.clusters[1]:
        plt.scatter(x, y, c='blue', s=7)

    for x, y in kms.clusters[2]:
        plt.scatter(x, y, c='green', s=7)

    plt.show()
    """


if __name__ == '__main__':
    main()
