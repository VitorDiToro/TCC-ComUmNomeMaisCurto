#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Author      :  Vitor Rodrigues Di Toro
# E-Mail      :  vitorrditoro@gmail.com
# Created On  :  19/05/2018
# Last Update :  31/05/2018

import os
import csv
import datetime
import statistics

from sources.dataSetUtils import DataSet
from sources.knn import KNN
from sources.distances import Distance, DistanceType

from sklearn import neighbors


def generate_csv(header: list, values: zip, filename: str, output_path: str="../outputs/"):

    date_time = datetime.datetime.now().strftime('%Y-%m-%d  %H.%M.%S')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_name = output_path + filename + date_time + ".csv"

    with open(file_name, 'w') as my_file:
        wr = csv.writer(my_file, lineterminator='\n')
        wr.writerow(header)

        for row in values:
            wr.writerow(row)


def skl_calculation_metrics(result_labels, test_labels):
    """

    :param result_labels:
    :param test_labels:
    :return:

    ref: http://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/
    """

    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0

    size = len(result_labels)
    for i in range(size):
        # Count true positives
        if result_labels[i] == 'g' and test_labels[i] == 'g':
            tp += 1.0

        # Count false positives
        if result_labels[i] == 'g' and test_labels[i] == 'b':
            fp += 1.0

        # Count true negatives
        if result_labels[i] == 'b' and test_labels[i] == 'b':
            tn += 1.0

        # Count false negatives
        if result_labels[i] == 'b' and test_labels[i] == 'g':
            fn += 1.0

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2.0 * (recall * precision) / (recall + precision)

    return accuracy, precision, recall, f1_score


def our_knn_experiment(k_first: int = 1, k_last: int = 350, times: int = 100,
                       distance_method: DistanceType = DistanceType.EUCLIDEAN,
                       data_set_path="data_set", output_path="../output/", verbose: bool = False):
    accuracy_values = []
    accuracy_mean = []
    accuracy_stdev = []

    precision_values = []
    precision_mean = []
    precision_stdev = []

    recall_values = []
    recall_mean = []
    recall_stdev = []

    f1_score_values = []
    f1_score_mean = []
    f1_score_stdev = []

    if k_first <= 0:
        k_first = 1

    if verbose:
        print("Calculating: ", end='')

    for k in range(k_first, k_last + 1):
        accuracy_values.clear()
        precision_values.clear()
        recall_values.clear()
        f1_score_values.clear()

        if verbose:
            if k == k_last:
                print("K" + str(k))
            else:
                print("K" + str(k) + ", ", end='')

        for i in range(times):

            training_group, test_group = DataSet.get_data(data_set_path, percent_to_training=60, randomize=True,
                                                          seed=i, verbose=False)
            knn = KNN(training_group, test_group)
            knn.fit(k=k, distance_method=distance_method)

            accuracy_values.append(knn.accuracy)
            precision_values.append(knn.precision)
            recall_values.append(knn.recall)
            f1_score_values.append(knn.f1_score)

        # Save results in CSV file
        accuracy_mean.append(statistics.mean(accuracy_values))
        accuracy_stdev.append(statistics.stdev(accuracy_values))
        precision_mean.append(statistics.mean(precision_values))
        precision_stdev.append(statistics.stdev(precision_values))
        recall_mean.append(statistics.mean(recall_values))
        recall_stdev.append(statistics.stdev(recall_values))
        f1_score_mean.append(statistics.mean(f1_score_values))
        f1_score_stdev.append(statistics.stdev(f1_score_values))

    # Save results in CSV file
    filename = "Our_-_" + distance_method.name() + "_k[" + str(k_first) + "_to_" + str(k_last) + "]_Times["\
               + str(times) + "]_-_"
    header = ["accuracy_mean", "accuracy_stdev",
              "precision_mean", "precision_stdev",
              "recall_mean", "recall_stdev",
              "f1_score_mean", "f1_score_stdev"]
    values = zip(* [accuracy_mean, accuracy_stdev,
                    precision_mean, precision_stdev,
                    recall_mean, recall_stdev,
                    f1_score_mean, f1_score_stdev])

    generate_csv(header, values, filename, output_path)


def skl_knn_experiment(k_first: int = 1, k_last: int = 350, times: int = 100,
                       distance_method: DistanceType = DistanceType.EUCLIDEAN,
                       data_set_path="data_set", output_path="../output/", p: float = 0.5, verbose: bool = False):
    accuracy_values = []
    accuracy_mean = []
    accuracy_stdev = []

    precision_values = []
    precision_mean = []
    precision_stdev = []

    recall_values = []
    recall_mean = []
    recall_stdev = []

    f1_score_values = []
    f1_score_mean = []
    f1_score_stdev = []

    if k_first <= 0:
        k_first = 1

    if verbose:
        print("Calculating: ", end='')

    for k in range(k_first, k_last + 1):
        accuracy_values.clear()
        precision_values.clear()
        recall_values.clear()
        f1_score_values.clear()

        if verbose:
            if k == k_last:
                print("K" + str(k))
            else:
                print("K" + str(k) + ", ", end='')

        for i in range(times):
            training_group, test_group = DataSet.get_data(data_set_path, percent_to_training=60, randomize=True, seed=i,
                                                          verbose=False)
            training_data = [t[:-1] for t in training_group]
            training_labels = [l[-1] for l in training_group]
            test_data = [t[:-1] for t in test_group]
            test_labels = [l[-1] for l in test_group]

            knn = neighbors.KNeighborsClassifier(n_neighbors=k, metric=distance_method.name(), algorithm='brute')
            knn.fit(training_data, training_labels)

            result_labels = knn.predict(test_data)
            result_labels = result_labels.tolist()

            accuracy, precision, recall, f1_score = skl_calculation_metrics(result_labels, test_labels)

            accuracy_values.append(accuracy)
            precision_values.append(precision)
            recall_values.append(recall)
            f1_score_values.append(f1_score)

        # Calculation Means ans StDevs
        accuracy_mean.append(statistics.mean(accuracy_values))
        accuracy_stdev.append(statistics.stdev(accuracy_values))
        precision_mean.append(statistics.mean(precision_values))
        precision_stdev.append(statistics.stdev(precision_values))
        recall_mean.append(statistics.mean(recall_values))
        recall_stdev.append(statistics.stdev(recall_values))
        f1_score_mean.append(statistics.mean(f1_score_values))
        f1_score_stdev.append(statistics.stdev(f1_score_values))

    # Save results in CSV file
    filename = "SKL_-_" + distance_method.name() + "_k[" + str(k_first) + "_to_" + str(k_last) + "]_Times["\
               + str(times) + "]_-_"
    header = ["accuracy_mean", "accuracy_stdev",
              "precision_mean", "precision_stdev",
              "recall_mean", "recall_stdev",
              "f1_score_mean", "f1_score_stdev"]
    values = zip(* [accuracy_mean, accuracy_stdev,
                    precision_mean, precision_stdev,
                    recall_mean, recall_stdev,
                    f1_score_mean, f1_score_stdev])

    generate_csv(header, values, filename, output_path)



def main():
    ds = DataSet()
    ds.fix_data_set('ionosphere', 'data')

    data_set_path = '../dataset/ionosphere.csv'
    output_path = "../outputs/knn/"

    k_first = 1
    k_last = 349
    times = 100

    our_knn_experiment(k_first, k_last, times, DistanceType.EUCLIDEAN, data_set_path, output_path, verbose=False)
    our_knn_experiment(k_first, k_last, times, DistanceType.MANHATTAN, data_set_path, output_path, verbose=False)
    our_knn_experiment(k_first, k_last, times, DistanceType.CHEBYSHEV, data_set_path, output_path, verbose=False)
    our_knn_experiment(k_first, k_last, times, DistanceType.MINKOWSKI, data_set_path, output_path, verbose=False)

    skl_knn_experiment(k_first, k_last, times, DistanceType.EUCLIDEAN, data_set_path, output_path, verbose=False)
    skl_knn_experiment(k_first, k_last, times, DistanceType.MANHATTAN, data_set_path, output_path, verbose=False)
    skl_knn_experiment(k_first, k_last, times, DistanceType.MINKOWSKI, data_set_path, output_path, verbose=False)
    skl_knn_experiment(k_first, k_last, times, DistanceType.CHEBYSHEV, data_set_path, output_path, verbose=False)


if __name__ == '__main__':
    main()
