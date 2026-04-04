from collections import namedtuple
from fractions import Fraction
from os.path import dirname, join
from random import uniform
import unittest

from data import one_word_data, small_data, small_uniform_data
from grams.grams import Gram, Histogram
from grams.utils import sample_size, map_to_binary


class HistogramTestSuite(unittest.TestCase):

    def test_histogram_correct_parent(self):
        self.assertEqual(Histogram.__bases__, (Gram,))

    def test_histogram_one_word(self):
        hist = Histogram("hello")
        self.assertDictEqual(hist.tokens_freqs, {"hello": 1})

    def test_histogram_one_word_with_tags(self):
        hist = Histogram("hello", use_pos_tags=True)
        self.assertDictEqual(hist.tokens_freqs, {("hello", "NN"): 1})

    def test_histogram_one_word_only_freqs(self):
        freqs = {"world": 1}
        hist = Histogram(tokens_freqs=freqs)
        self.assertDictEqual(hist.tokens_freqs, {"world": 1})

    @unittest.skip("Implement corpus table consolidation")
    def test_histogram_combined_freqs(self):
        freqs = {"world": 1}
        hist = Histogram("hello", tokens_freqs=freqs)
        self.assertDictEqual(hist.tokens_freqs, {"hello": 1, "world": 1})