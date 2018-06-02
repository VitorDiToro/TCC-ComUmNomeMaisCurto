#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Author        :   Vitor Rodrigues Di Toro
# E-Mail        :   vitorrditoro@gmail.com
# Created on    :   14/03/2018
# Last Update   :   31/05/2018

import unittest
from enum import Enum


class DistanceType(Enum):
    EUCLIDEAN = 'euclidean'
    MANHATTAN = 'manhattan'
    MINKOWSKI = 'minkowski'
    CHEBYSHEV = 'chebyshev'

    def name(self):
        return self._value_


class Distance:
    type: DistanceType

    def __init__(self):
        self.distance_order = None
        self.type = None

    def set_distance_order(self, distance_order):
        self.distance_order = distance_order

    def calculator(self, p1, p2, distance_method: DistanceType):

        if distance_method == DistanceType.EUCLIDEAN:
            """
            Euclidean Distance implementation
                
            See: https://en.wikipedia.org/wiki/Euclidean_distance
            """
            sum_value = 0

            for u, v in zip(p1, p2):
                try:
                    sum_value += (u - v) ** 2
                except TypeError:
                    pass

            result = sum_value ** 0.5

        elif distance_method == DistanceType.MANHATTAN:
            """
            Manhattan Distance implementation
        
            See: https://en.wikipedia.org/wiki/Taxicab_geometry
            """

            sum_value = 0

            for u, v in zip(p1, p2):
                try:
                    sum_value += abs(u - v)
                except TypeError:
                    pass

            result = sum_value

        elif distance_method == DistanceType.MINKOWSKI:
            """
            Minkowski Distance implementation
    
            See: http://en.wikipedia.org/wiki/Minkowski_distance
            """

            sum_value = 0

            for u, v in zip(p1, p2):
                try:
                    sum_value += abs(u - v) ** self.distance_order
                except TypeError:
                    pass

            result = sum_value ** (1 / self.distance_order)

        elif distance_method == DistanceType.CHEBYSHEV:
            """        
        
            See: https://en.wikipedia.org/wiki/Chebyshev_distance
            """

            max_distance = -1

            for u, v in zip(p1, p2):
                try:
                    axis_distance = abs(u - v)
                    max_distance = max(max_distance, axis_distance)
                except TypeError:
                    pass

            result = max_distance

        else:
            raise ValueError("Algo de errado não está certo!")

        return result


class TestDistances(unittest.TestCase):

    def test_euclidean_distance(self):
        # ref: http://calculator.vhex.net/calculator/distance/euclidean-distance

        distance = Distance()

        p1 = [0, 0, 0, 'b']
        p2 = [2, 2, 2, 'g']
        self.assertEqual(distance.calculator(p1, p2, DistanceType.EUCLIDEAN), (2 * (3 ** 0.5)))

        p1 = [0, 5, -8, 9, 3, 'b']
        p2 = [2, 7, -9, -1, 3, 'b']
        self.assertEqual(round(distance.calculator(p1, p2, DistanceType.EUCLIDEAN), 6), 10.440307)

        p1 = [0.500, 5.30, -8.2, 9.54, 3.134]
        p2 = [2.123, 7.43, -9.2, -1.50, 3.000]
        self.assertEqual(round(distance.calculator(p1, p2, DistanceType.EUCLIDEAN), 6), 11.404849)

    def test_manhattan_distance(self):
        # ref: http://calculator.vhex.net/calculator/distance/manhattan-distance

        distance = Distance()

        p1 = [0, 0, 0, "String"]
        p2 = [2, 2, 2, "Str"]
        self.assertEqual(distance.calculator(p1, p2, DistanceType.MANHATTAN), 6)

        p1 = [0, 5, -8, 9, 3]
        p2 = [2, 7, -9, -1, 3]
        self.assertEqual(distance.calculator(p1, p2, DistanceType.MANHATTAN), 15)

        p1 = [0.500, 5.30, -8.2, 9.54, 3.134]
        p2 = [2.123, 7.43, -9.2, -1.50, 3.000]
        self.assertEqual(distance.calculator(p1, p2, DistanceType.MANHATTAN), 15.927)

    def test_minkowski_distance(self):
        # ref: http://people.revoledu.com/kardi/tutorial/Similarity/MinkowskiDistance.html

        distance = Distance()

        p1 = [0, 3, 4, 5, 'g']
        p2 = [7, 6, 3, -1, 'b']
        n = 3
        distance.set_distance_order(n)
        self.assertEqual(round(distance.calculator(p1, p2, DistanceType.MINKOWSKI), 3), 8.373)

        p1 = [0.5, 3.9, -4.37, 5.5, "Str"]
        p2 = [7.72, 6.36, 3.27, -1.98, "Sting"]
        n = -0.25
        distance.set_distance_order(n)
        self.assertEqual(round(distance.calculator(p1, p2, DistanceType.MINKOWSKI), 5), 0.02139)

        p1 = [0.5, 3.9, -4.37, 5.5]
        p2 = [7.72, 6.36, 3.27, -1.98]
        n = 4
        distance.set_distance_order(n)
        self.assertEqual(round(distance.calculator(p1, p2, DistanceType.MINKOWSKI), 3), 9.818)

    def test_distance_types(self):
        # Euclidean Distance
        self.assertEqual(DistanceType.EUCLIDEAN.name(), 'euclidean')
        # Manhattan Distance
        self.assertEqual(DistanceType.MANHATTAN.name(), 'manhattan')
        # Minkowski Distance
        self.assertEqual(DistanceType.MINKOWSKI.name(), 'minkowski')
        # Chebyshev Distance
        self.assertEqual(DistanceType.CHEBYSHEV.name(), 'chebyshev')


if __name__ == "__main__":
    unittest.main()
