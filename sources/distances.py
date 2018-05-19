#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Author        :   Vitor Rodrigues Di Toro
# E-Mail        :   vitorrditoro@gmail.com
# Date          :   14/03/2018
# Last Update   :   09/05/2018
#

import unittest


class DistanceType(int):
    euclidean = 1
    manhattan = 2
    minkowski = 3

    def __init__(self, v):
        super().__init__(v)
        pass


class Distance:

    def __init__(self):
        self.distance_order = None

    class Type:
        euclidean = DistanceType(1)
        manhattan = DistanceType(2)
        minkowski = DistanceType(3)

    def set_distance_order(self, distance_order):
        self.distance_order = distance_order

    def calculator(self, p1, p2, distance_method: DistanceType):

        if distance_method == DistanceType.euclidean:
            """
            EuclideanDistance Distance implementation
                
            See: https://en.wikipedia.org/wiki/Euclidean_distance
            """
            sum_value = 0

            for u, v in zip(p1, p2):
                try:
                    sum_value += (u - v) ** 2
                except TypeError:
                    pass

            result = sum_value ** 0.5

        elif distance_method == DistanceType.manhattan:
            """
            ManhattanDistance Distance implementation
        
            See: https://en.wikipedia.org/wiki/Taxicab_geometry
            """

            sum_value = 0

            for u, v in zip(p1, p2):
                try:
                    sum_value += abs(u - v)
                except TypeError:
                    pass

            result = sum_value

        elif distance_method == DistanceType.minkowski:
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

        else:
            result = None

        return result


class TestDistances(unittest.TestCase):

    def test_euclidean_distance(self):
        # ref: http://calculator.vhex.net/calculator/distance/euclidean-distance

        distance = Distance()

        p1 = [0, 0, 0, 'b']
        p2 = [2, 2, 2, 'g']

        self.assertEqual(distance.calculator(p1, p2, Distance.Type.euclidean), (2 * (3 ** 0.5)))

        p1 = [0, 5, -8, 9, 3, 'b']
        p2 = [2, 7, -9, -1, 3, 'b']
        self.assertEqual(round(distance.calculator(p1, p2, Distance.Type.euclidean), 6), 10.440307)

        p1 = [0.500, 5.30, -8.2, 9.54, 3.134]
        p2 = [2.123, 7.43, -9.2, -1.50, 3.000]
        self.assertEqual(round(distance.calculator(p1, p2, Distance.Type.euclidean), 6), 11.404849)

    def test_manhattan_distance(self):
        # ref: http://calculator.vhex.net/calculator/distance/manhattan-distance

        distance = Distance()

        p1 = [0, 0, 0, "String"]
        p2 = [2, 2, 2, "Str"]

        self.assertEqual(distance.calculator(p1, p2, Distance.Type.manhattan), 6)

        p1 = [0, 5, -8, 9, 3]
        p2 = [2, 7, -9, -1, 3]
        self.assertEqual(distance.calculator(p1, p2, Distance.Type.manhattan), 15)

        p1 = [0.500, 5.30, -8.2, 9.54, 3.134]
        p2 = [2.123, 7.43, -9.2, -1.50, 3.000]
        self.assertEqual(distance.calculator(p1, p2, Distance.Type.manhattan), 15.927)

    def test_minkowski_distance(self):
        # ref: http://people.revoledu.com/kardi/tutorial/Similarity/MinkowskiDistance.html

        distance = Distance()

        p1 = [0, 3, 4, 5, 'g']
        p2 = [7, 6, 3, -1, 'b']
        n = 3

        distance.set_distance_order(n)
        self.assertEqual(round(distance.calculator(p1, p2, Distance.Type.minkowski), 3), 8.373)

        p1 = [0.5, 3.9, -4.37, 5.5, "Str"]
        p2 = [7.72, 6.36, 3.27, -1.98, "Sting"]
        n = -0.25

        distance.set_distance_order(n)
        self.assertEqual(round(distance.calculator(p1, p2, Distance.Type.minkowski), 5), 0.02139)

        p1 = [0.5, 3.9, -4.37, 5.5]
        p2 = [7.72, 6.36, 3.27, -1.98]
        n = 4

        distance.set_distance_order(n)
        self.assertEqual(round(distance.calculator(p1, p2, Distance.Type.minkowski), 3), 9.818)

    def test_distance_types(self):
        # Euclidean Distance
        self.assertLessEqual(Distance.Type.euclidean, 1)
        # Manhattan Distance
        self.assertLessEqual(Distance.Type.manhattan, 2)
        # Minkowski Distance
        self.assertLessEqual(Distance.Type.minkowski, 3)


if __name__ == "__main__":
    unittest.main()
