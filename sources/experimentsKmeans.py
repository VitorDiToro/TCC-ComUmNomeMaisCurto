# -*- coding: utf-8 -*-
"""
Created on Sat May 19 08:52:36 2018

authors: Vitor Rodrigues Di Toro
         Marcelo Vinicios Cysneiros Aragão
         Jonatan Alberto Afonso
"""

import os
import statistics

from sources.distances import DistanceType
from sources.dataSetUtils import DataSet, generate_csv
from sources.kmeans import KMeans as own_Kmeans

from sklearn.cluster import KMeans as skl_Kmeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score


def run_our_implementation(data, k: int, seed=0):
    own_kms = own_Kmeans(k=k, tolerance=1e-10, max_iterations=5000)
    own_kms.fit_predict(data, distance_method=DistanceType.EUCLIDEAN, seed=seed)

    n_iterations = own_kms.iteration
    silhouette = silhouette_score(data, own_kms.labels, metric='euclidean', random_state=0)
    calinski_harabaz = calinski_harabaz_score(data, own_kms.labels)

    return n_iterations, silhouette, calinski_harabaz


def run_skl_implementation(data, k: int, seed=0):

    skl_kms = skl_Kmeans(n_clusters=k, algorithm='auto', tol=1e-10, max_iter=5000, random_state=seed)
    skl_kms.fit_predict(data)

    n_iterations = skl_kms.n_iter_
    silhouette = silhouette_score(data, skl_kms.labels_, metric='euclidean', random_state=0)
    calinski_harabaz = calinski_harabaz_score(data, skl_kms.labels_)

    return n_iterations, silhouette, calinski_harabaz


def run(times, k_first, k_last, output_path):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # statistics for our own implementation
    n_iterations_our = []
    mean_n_iterations_our = []
    stdev_n_iterations_our = []

    silhouette_scores_our = []
    mean_silhouette_scores_our = []
    stdev_silhouette_scores_our = []

    calinski_harabaz_scores_our = []
    mean_calinski_harabaz_scores_our = []
    stdev_calinski_harabaz_scores_our = []

    # statistics for SKL implementation
    n_iterations_skl = []
    mean_n_iterations_skl = []
    stdev_n_iterations_skl = []

    silhouette_scores_skl = []
    mean_silhouette_scores_skl = []
    stdev_silhouette_scores_skl = []

    calinski_harabaz_scores_skl = []
    mean_calinski_harabaz_scores_skl = []
    stdev_calinski_harabaz_scores_skl = []

    for k in range(k_first, k_last + 1):
        for i in range(times):

            # TODO read only once ; randomize 'times' times

            data = DataSet.get_data_lc('../dataset/ionosphere.csv', range(350), range(34), randomize=True, seed=i)

            # run our own implementation
            n_iterations, silhouette, calinski_harabaz = run_our_implementation(data, k=k, seed=i)

            n_iterations_our.append(n_iterations)
            silhouette_scores_our.append(silhouette)
            calinski_harabaz_scores_our.append(calinski_harabaz)

            # run SKL implementation
            n_iterations, silhouette, calinski_harabaz = run_skl_implementation(data, k=k, seed=i)

            n_iterations_skl.append(n_iterations)
            silhouette_scores_skl.append(silhouette)
            calinski_harabaz_scores_skl.append(calinski_harabaz)

        # calculate statistics for our own implementation
        mean_n_iterations_our.append(statistics.mean(n_iterations_our))
        stdev_n_iterations_our.append(statistics.stdev(n_iterations_our))
        mean_silhouette_scores_our.append(statistics.mean(silhouette_scores_our))
        stdev_silhouette_scores_our.append(statistics.stdev(silhouette_scores_our))
        mean_calinski_harabaz_scores_our.append(statistics.mean(calinski_harabaz_scores_our))
        stdev_calinski_harabaz_scores_our.append(statistics.stdev(calinski_harabaz_scores_our))

        # calculate statistics for SKL implementation
        mean_n_iterations_skl.append(statistics.mean(n_iterations_skl))
        stdev_n_iterations_skl.append(statistics.stdev(n_iterations_skl))
        mean_silhouette_scores_skl.append(statistics.mean(silhouette_scores_skl))
        stdev_silhouette_scores_skl.append(statistics.stdev(silhouette_scores_skl))
        mean_calinski_harabaz_scores_skl.append(statistics.mean(calinski_harabaz_scores_skl))
        stdev_calinski_harabaz_scores_skl.append(statistics.stdev(calinski_harabaz_scores_skl))

    # CSV header
    header = ["k",
              "iteration_mean", "iteration_stdev",
              "silhouette_scores_mean", "silhouette_scores_stdev",
              "calinski_harabaz_scores_mean", "calinski_harabaz_scores_stdev"]

    # persist statistics for our own implementation
    file_name = "Our_Implementation_-_Times[" + str(times) + "]_-_"
    values = zip(*[[k for k in range(k_first, k_last+1)],
                   mean_n_iterations_our, stdev_n_iterations_our,
                   mean_silhouette_scores_our, stdev_silhouette_scores_our,
                   mean_calinski_harabaz_scores_our, stdev_calinski_harabaz_scores_our])
    generate_csv(header, values, file_name, output_path)

    # persist statistics for SKL implementation
    file_name = "SKL_Implementation_-_Times[" + str(times) + "]_-_"
    values = zip(*[[k for k in range(k_first, k_last+1)],
                   mean_n_iterations_skl, stdev_n_iterations_skl,
                   mean_silhouette_scores_skl, stdev_silhouette_scores_skl,
                   mean_calinski_harabaz_scores_skl, stdev_calinski_harabaz_scores_skl])
    generate_csv(header, values, file_name, output_path)


def main():
    times = 100
    output_path = "../outputs/kmeans/"
    k_firs = 2
    k_last = 5

    run(times, k_firs, k_last, output_path)


if __name__ == '__main__':
    main()
