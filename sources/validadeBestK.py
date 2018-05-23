#!/usr/bin/python3
# -*- coding: utf-8 -*-

# __main__.py

# Author      :  Vitor Rodrigues Di Toro
# E-Mail      :  vitorrditoro@gmail.com
# Create      :  19/05/2018
# Last Update :  23/05/2018

import sys
import csv
import datetime
import matplotlib.pyplot as plt
import statistics

sys.path.append('../')

from sources.dataSetUtils import *
from sources.knn import KNN
from sources.distances import *


def generate_csv(mean, stdev, k_first, k_last, times, distance_method):
    import csv
    import datetime

    date_now = datetime.datetime.now().strftime('%Y-%m-%d  %H.%M.%S')
    dist_method_str = Distance.Type.get_str(distance_method)

    file_name = dist_method_str + "_k[" + str(k_first) + "_to_" + str(k_last) + "]_" + str(times) + \
                "Times_-_" + date_now + ".csv"

    with open(file_name, 'w') as myfile:
        wr = csv.writer(myfile, lineterminator='\n')  # , quoting=csv.QUOTE_ALL)
        wr.writerow(["mean_accuracy", "stdev_accuracy", "mean_recall", "stdev_recall",
                     "mean_sf1", "stdev_sf1"])

        for row in zip(mean, stdev):
            wr.writerow([float(row[0]), float(row[1])])


def test_k_and_save_csv(k_first: int = 1, k_last: int = 350, times: int = 100,
                        distance_method: Distance.Type = Distance.Type.euclidean, verbose: bool = False):
    if k_first <= 0:
        k_first = 1

    ds = DataSet()
    ds.fix_data_set('ionosphere', 'data')

    data_set_name = '../dataset/ionosphere.csv'

    mean_accuracy = []
    stdev_accuracy = []
    # recall_mean = []
    # recall_stdev = []
    # sf1_mean = []
    # sf1_stdev = []

    print("Calculating: ", end='')
    for k in range(k_first, k_last + 1):
        values = []

        if k == k_last:
            print("K" + str(k))
        else:
            print("K" + str(k) + ", ", end='')

        for i in range(times):
            training_data, test_data = DataSet.get_data(data_set_name, percent_to_training=60,
                                                        randomize=True, verbose=False)
            knn = KNN(training_data, test_data)
            knn.fit(k=k, distance_method=distance_method)

            values.append(knn.accuracy)
            if verbose:
                print("accuracy: " + str(knn.accuracy))

        mean_accuracy.append(sum(values) / len(values))
        stdev_accuracy.append(statistics.stdev(values))

    generate_csv(mean_accuracy, stdev_accuracy, k_first, k_last, times, distance_method)


def main():

    k_first = 1
    k_last = 10
    times = 100

    test_k_and_save_csv(k_first, k_last, times, Distance.Type.euclidean(), verbose=False)
    test_k_and_save_csv(k_first, k_last, times, Distance.Type.manhattan(), verbose=False)
    test_k_and_save_csv(k_first, k_last, times, Distance.Type.minkowski(), verbose=False)


if __name__ == '__main__':
    main()
