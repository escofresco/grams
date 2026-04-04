from collections import namedtuple
from fractions import Fraction
from random import randrange
import unittest

import numpy as np

from grams.online import Avg, Var


class OnlineVarianceTestSuite(unittest.TestCase):

    def test_online_variance_uniform_decimal(self):
        variance = Var()
        array = tuple(Fraction(randrange(1001), 100)
                      for _ in range(100000))  # variance =~ 1/12, mean =~ 1/2
        array_as_floats = tuple(map(float, array))
        expected_stats = self.expected_stats(array_as_floats)
        for num in array:
            variance.add(num)
        self.make_online_variance_asserts(expected_stats, variance)

    def test_online_variance_uniform_natural(self):
        variance = Var()
        array = tuple(randrange(1001) for _ in range(100000))
        expected_stats = self.expected_stats(array)
        for num in array:
            variance.add(num)
        self.make_online_variance_asserts(expected_stats, variance)

    @unittest.skip("Remove will likely get removed.")
    def test_online_variance_uniform_remove(self):
        variance = Var()
        array = [Fraction(randrange(101), 100) for _ in range(1000)]
        array_as_floats = list(map(float, array))
        for num in array:
            variance.add(num)

        for i in range(1000, -1, -100):
            for _ in range(100):
                variance.remove(array.pop())
            expected_stats = self.expected_stats(array_as_floats)
            with self.subTest(i=i):
                print(i)
                self.make_online_variance_asserts(expected_stats, variance)

    def test_norm_exclusively_removes(self):
        pass

    def make_online_variance_asserts(self, expected_stats, variance):
        self.assertAlmostEqual(expected_stats.popvar, float(variance.var()), 3)
        self.assertAlmostEqual(expected_stats.samplevar,
                               float(variance.var(ddof=1)))
        self.assertAlmostEqual(expected_stats.popstd, float(variance.std()), 3)
        self.assertAlmostEqual(expected_stats.samplestd,
                               float(variance.std(ddof=1)))
        self.assertAlmostEqual(expected_stats.mean, float(variance.mean))

    @staticmethod
    def expected_stats(array):
        Expected = namedtuple("Expected",
                              "popvar samplevar mean popstd samplestd ")
        return Expected(np.var(array), np.var(array, ddof=1), np.mean(array),
                        np.std(array), np.std(array, ddof=1))


class OnlineAverageTestSuite(unittest.TestCase):

    def setUp(self):
        self.online_avg = Avg()
        self.mono_inc_array = list(range(1, 100001,
                                         7))  # monotonic increasing every 7
        self.mono_bounds_large_array = [randrange(i) for i in range(1, 100001)
                                       ]  # monotonic bounds-increasing
        self.normal_large_array = [randrange(10000) for _ in range(100000)
                                  ]  # normal distribution

    def test_mono_inc_exclusively_adds(self):
        running_sum = 0

        for i, val in enumerate(self.mono_inc_array):
            self.online_avg.add(val)
            running_sum += val
            expected_avg = running_sum / (i + 1)
            with self.subTest(i=i):
                self.assertAlmostEqual(expected_avg,
                                       float(self.online_avg),
                                       places=5)
        expected_final = sum(self.mono_inc_array) / len(self.mono_inc_array)
        self.assertAlmostEqual(expected_final, float(self.online_avg), places=5)

    def test_mono_exclusively_adds(self):
        running_sum = 0

        for i, val in enumerate(self.mono_bounds_large_array):
            self.online_avg.add(val)
            running_sum += val
            expected_avg = running_sum / (i + 1)
            with self.subTest(i=i):
                self.assertAlmostEqual(expected_avg,
                                       float(self.online_avg),
                                       places=5)
        expected_final = sum(self.mono_bounds_large_array) / len(
            self.mono_bounds_large_array)
        self.assertAlmostEqual(expected_final, float(self.online_avg), places=5)

    def test_norm_exclusively_adds(self):
        running_sum = 0

        for i, val in enumerate(self.normal_large_array):
            self.online_avg.add(val)
            running_sum += val
            expected_avg = running_sum / (i + 1)
            with self.subTest(i=i):
                self.assertAlmostEqual(expected_avg,
                                       float(self.online_avg),
                                       places=5)
        expected_final = sum(self.normal_large_array) / len(
            self.normal_large_array)
        self.assertAlmostEqual(expected_final, float(self.online_avg), places=5)

    def test_empty_average(self):
        """Ensure the class handles being evaluated before any data is added."""
        empty_avg = Avg()
        # If your design dictates it should return 0.0:
        self.assertEqual(0.0, float(empty_avg))

        # OR, if your design dictates it SHOULD raise an error:
        # with self.assertRaises(ZeroDivisionError):
        #     float(empty_avg)

    def test_mixed_signs_and_zeroes(self):
        """Test combinations of positive, negative, and zero values."""
        array = [100, -50, 0, 0, -1000, 500, -50]
        running_sum = 0

        for i, val in enumerate(array):
            self.online_avg.add(val)
            running_sum += val
            expected_avg = running_sum / (i + 1)
            with self.subTest(i=i, val=val):
                self.assertAlmostEqual(expected_avg,
                                       float(self.online_avg),
                                       places=5)

    def test_extreme_floating_point_precision(self):
        """
        Catastrophic cancellation test. 
        A naive sum() will lose the 1.0s due to floating point limits.
        """
        # 1 quadrillion, negative 1 quadrillion, then three 1s.
        # The true average is exactly 0.6
        array = [1e15, -1e15, 1.0, 1.0, 1.0]
        running_sum = 0

        for i, val in enumerate(array):
            self.online_avg.add(val)
            running_sum += val
            expected_avg = running_sum / (i + 1)
            with self.subTest(i=i):
                self.assertAlmostEqual(expected_avg,
                                       float(self.online_avg),
                                       places=5)

    def test_fraction_inputs(self):
        """Ensure the average handles the Fraction data type cleanly."""
        array = [Fraction(1, 3), Fraction(1, 2), Fraction(2, 3)]
        # Expected average of (1/3 + 1/2 + 2/3) is 1/2

        for val in array:
            self.online_avg.add(val)

        self.assertAlmostEqual(0.5, float(self.online_avg), places=5)
