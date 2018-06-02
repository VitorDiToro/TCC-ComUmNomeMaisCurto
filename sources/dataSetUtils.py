#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Implementação do CU DA SUA MÃE!
#   Brincadeira, é a implementação de funções úteis p/ manipular o DataSet.

# Author        :   Vitor Rodrigues Di Toro
# E-Mail        :   vitorrditoro@gmail.com
# Created on    :   23/03/2018
# Last Update   :   19/05/2018

import csv
import datetime
import os
import random


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


def num(s):
    # TODO --> Fix DocString
    """

    :param s:
    :return:
    """
    try:
        return float(s)
    except ValueError:
        return s


class DataSet:
    @staticmethod
    def fix_data_set(data_set_name, extension):
        # TODO --> Fix DocString
        """

        :param data_set_name:
        :param extension:
        :return:
        """

        with open('../dataset/' + data_set_name + "." + extension, 'r') as File:
            reader = csv.reader(File, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
            l_reader = list(reader)

        my_csv_list = []
        for i in range(len(l_reader)):
            if not(l_reader[i] in l_reader[i+1:len(l_reader)]):
                my_csv_list.append(l_reader[i])

        with open("../dataset/" + data_set_name + ".csv", "w") as File:
            writer = csv.writer(File, lineterminator='\n')
            writer.writerows(my_csv_list)

    @staticmethod
    def get_data_lc(data_set_name, lines, columns, seed=0, randomize=False):
        # TODO --> Fix DocString
        """

        :param data_set_name:
        :param lines:
        :param columns:
        :param seed:
        :param randomize:
        :return:
        """
        with open(data_set_name, 'r') as File:
            reader = csv.reader(File, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
            l_reader = list(reader)

            if randomize:
                random.seed(seed)
                random.shuffle(l_reader)

        data = []
        tmp = [[l_reader[i][j] for j in columns] for i in lines]
        for row in tmp:
            data.append([num(i) for i in row])

        return data

    @staticmethod
    def get_data(dataset_name: str='data.csv', percent_to_training: int=60, randomize: bool=True, seed: int=0,
                 verbose: bool=True):
        # TODO --> Fix DocString
        """
        :param dataset_name:
        :param percent_to_training:
        :param randomize:
        :param seed:
        :param verbose:
        :return:
        """
        count = 0
        group_g = 0
        test_data = []
        training_data = []

        with open(dataset_name, 'r') as File:
            reader = csv.reader(File, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)

            l_reader = list(reader)
            limit = int(len(l_reader) * (percent_to_training / 100))

            if randomize:
                random.seed(seed)
                random.shuffle(l_reader)

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
