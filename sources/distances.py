#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Author        :   Vitor Rodrigues Di Toro
# E-Mail        :   vitorrditoro@gmail.com
# Date          :   14/03/2018
# Last Update   :   04/05/2018
#

import unittest


class DistanceCalculator:

    def __init__(self):
        pass

    @staticmethod
    def euclidean_distance(p1, p2):
        """
        EuclideanDistance Distance implementation

        :param p1: first array
        :param p2: second array
        :return: EuclideanDistance distance between two vectors

        See: https://en.wikipedia.org/wiki/Euclidean_distance
        """

        sum = 0

        for u, v in zip(p1, p2):
            try:
                sum += (u - v) ** 2
            except:
                pass

        return sum ** (0.5)

    @staticmethod
    def manhattan_distance(p1, p2):
        """
        ManhattanDistance Distance implementation

        :param p1: first array
        :param p2: second array
        :return: ManhattanDistance distance between two vectors.

        See: https://en.wikipedia.org/wiki/Taxicab_geometry
        """

        sum = 0

        for u, v in zip(p1, p2):
            try:
                sum += abs(u - v)
            except:
                pass

        return sum

    @staticmethod
    def minkowski_distance(p1, p2, n):
        """
        Minkowski Distance implementation

        :param p1: first array
        :param p2: second array
        :param n: distance order
        :return: Minkowski distance between two vectors.

        See: http://en.wikipedia.org/wiki/Minkowski_distance
        """

        sum = 0

        for u, v in zip(p1, p2):
            try:
                sum += abs(u - v) ** n
            except:
                pass

        return sum ** (1 / n)


class TestDistances(unittest.TestCase):

    def test_euclidean_distance(self):
        # ref: http://calculator.vhex.net/calculator/distance/euclidean-distance

        p1 = [0, 0, 0, 'b']
        p2 = [2, 2, 2, 'g']
        self.assertEqual(DistanceCalculator.euclidean_distance(p1, p2), (2 * (3) ** (0.5)))

        p1 = [0, 5, -8, 9, 3, 'b']
        p2 = [2, 7, -9, -1, 3, 'b']
        self.assertEqual(round(DistanceCalculator.euclidean_distance(p1, p2), 6), 10.440307)

        p1 = [0.500, 5.30, -8.2, 9.54, 3.134]
        p2 = [2.123, 7.43, -9.2, -1.50, 3.000]
        self.assertEqual(round(DistanceCalculator.euclidean_distance(p1, p2), 6), 11.404849)

    def test_manhattan_distance(self):
        # ref: http://calculator.vhex.net/calculator/distance/manhattan-distance

        p1 = [0, 0, 0, "String"]
        p2 = [2, 2, 2, "Str"]
        self.assertEqual(DistanceCalculator.manhattan_distance(p1, p2), 6)

        p1 = [0, 5, -8, 9, 3]
        p2 = [2, 7, -9, -1, 3]
        self.assertEqual(DistanceCalculator.manhattan_distance(p1, p2), 15)

        p1 = [0.500, 5.30, -8.2, 9.54, 3.134]
        p2 = [2.123, 7.43, -9.2, -1.50, 3.000]
        self.assertEqual(DistanceCalculator.manhattan_distance(p1, p2), 15.927)

    def test_minkowski_distance(self):
        # ref: http://people.revoledu.com/kardi/tutorial/Similarity/MinkowskiDistance.html

        p1 = [0, 3, 4, 5, 'g']
        p2 = [7, 6, 3, -1, 'b']
        n = 3
        self.assertEqual(round(DistanceCalculator.minkowski_distance(p1, p2, n), 3), 8.373)

        p1 = [0.5, 3.9, -4.37, 5.5, "Str"]
        p2 = [7.72, 6.36, 3.27, -1.98, "Sting"]
        n = -0.25
        self.assertEqual(round(DistanceCalculator.minkowski_distance(p1, p2, n), 5), 0.02139)

        p1 = [0.5, 3.9, -4.37, 5.5]
        p2 = [7.72, 6.36, 3.27, -1.98]
        n = 4
        self.assertEqual(round(DistanceCalculator.minkowski_distance(p1, p2, n), 3), 9.818)


if __name__ == "__main__":
    unittest.main()
