#!/usr/bin/python3
# -*- coding: utf-8 -*-

# __main__.py

# Author      :  Vitor Rodrigues Di Toro
# E-Mail      :  vitorrditoro@gmail.com
# Create      :  19/05/2018
# Last Update :  24/05/2018

import sys
import csv
import datetime
import statistics

sys.path.append('../')

from sources.dataSetUtils import *
from sources.knn import KNN
from sources.distances import *


def generate_csv(accuracy_mean, accuracy_stdev,
                 recall_mean, recall_stdev,
                 f1_score_mean, f1_score_stdev,
                 k_first, k_last, times, distance_method):

    date_now = datetime.datetime.now().strftime('%Y-%m-%d  %H.%M.%S')
    dist_method_str = Distance.Type.get_str(distance_method)
    path = "../outputs/"

    file_name = path + dist_method_str + "_k[" + str(k_first) + "_to_" + str(k_last) + "]_Times[" + str(times) + "]_-_" + \
                date_now + ".csv"

    with open(file_name, 'w') as myfile:
        wr = csv.writer(myfile, lineterminator='\n')  # , quoting=csv.QUOTE_ALL)
        wr.writerow(["mean_accuracy", "stdev_accuracy", "mean_recall", "stdev_recall",
                     "mean_sf1", "stdev_sf1"])

        for row in zip(accuracy_mean, accuracy_stdev, recall_mean, recall_stdev, f1_score_mean, f1_score_stdev):
            wr.writerow([row[0], row[1], row[2], row[3], row[4], row[5]])


def test_k_and_save_csv(k_first: int = 1, k_last: int = 350, times: int = 100,
                        distance_method: Distance.Type = Distance.Type.euclidean, verbose: bool = False):
    if k_first <= 0:
        k_first = 1

    ds = DataSet()
    ds.fix_data_set('ionosphere', 'data')

    data_set_name = '../dataset/ionosphere.csv'

    accuracy_mean = []
    accuracy_stdev = []
    recall_mean = []
    recall_stdev = []
    f1_score_mean = []
    f1_score_stdev = []

    if verbose:
        print("Calculating: ", end='')

    for k in range(k_first, k_last + 1):
        accuracy_values = []
        recall_values = []
        f1_score_values = []

        if verbose:
            if k == k_last:
                print("K" + str(k))
            else:
                print("K" + str(k) + ", ", end='')

        for i in range(times):
            training_data, test_data = DataSet.get_data(data_set_name, percent_to_training=60, randomize=True,
                                                        verbose=False)
            knn = KNN(training_data, test_data)
            knn.fit(k=k, distance_method=distance_method)

            accuracy_values.append(knn.accuracy)
            recall_values.append(knn.recall)
            f1_score_values.append(knn.f1_score)

            if verbose:
                print("accuracy: " + str(knn.accuracy))

        accuracy_mean.append(sum(accuracy_values) / len(accuracy_values))
        accuracy_stdev.append(statistics.stdev(accuracy_values))

        recall_mean.append(sum(recall_values) / len(recall_values))
        recall_stdev.append(statistics.stdev(recall_values))

        f1_score_mean.append(sum(f1_score_values) / len(f1_score_values))
        f1_score_stdev.append(statistics.stdev(f1_score_values))

    generate_csv(accuracy_mean, accuracy_stdev,
                 recall_mean, recall_stdev,
                 f1_score_mean, f1_score_stdev,
                 k_first, k_last, times, distance_method)


def main():
    k_first = 1
    k_last = 5
    times = 5

    #test_k_and_save_csv(k_first, k_last, times, Distance.Type.euclidean(), verbose=False)
    test_k_and_save_csv(k_first, k_last, times, Distance.Type.manhattan(), verbose=False)
    #test_k_and_save_csv(k_first, k_last, times, Distance.Type.minkowski(), verbose=False)


if __name__ == '__main__':
    main()
