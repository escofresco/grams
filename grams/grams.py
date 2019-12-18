#!python

from __future__ import division, print_function  # Python 2 and 3 compatibility
from array import array
from collections import Counter, defaultdict, deque, namedtuple
from dataclasses import make_dataclass
from fractions import Fraction
from functools import wraps
from os import devnull
from random import random, choice
import sys
from typing import Iterable, Optional, Tuple, Union

from coverage import Coverage, CoverageData

from .online import Var
from .stats import FreqDist, Sample
from .termgraph import showgraph
from .utils import (binsearch, invert_dict, LogMethodCalls,
                    merge_data_containing_ints, p)

__all__ = ["Histogram", "Listogram", "Dictogram"]


class Gram(FreqDist):  #, metaclass=LogMethodCalls, logs_size=4):
    """This is a generic histogram which holds a distribution of data, which
    have been clumped into bins by the parent class, FreqDist. Bins are aligned
    in two dimensions, data and frequency, where frequency is calculated from
    data.

    Args:
        data: This represents the input to a probability distribution function.
    """
    __slots__ = ("data_frequency")

    def __init__(self, data_frequency):
        super().__init__(data_frequency)
        self.sampler = Sample(self)

    def similarity(self, other):
        return super().similarity(other)

    def frequency(self, word):
        raise NotImplementedError()

    def sample(self):
        return self.sampler.randword()

    @staticmethod
    def bin_search(array, prob):
        """Search for the lowest matching probability in array consisting of
        namedtuples. this is the sample component of roulette wheel sampling"""
        lo = 0
        hi = len(array) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if (array[mid].cumulative <= prob and
                (mid == len(array) - 1 or array[mid + 1].cumulative > prob)):
                return mid
            if array[mid].cumulative < prob:
                lo = mid + 1
            else:
                hi = mid
        return None


class Histogram(Gram, metaclass=LogMethodCalls, logs_size=4):

    __slots__ = ("source_text", "word_to_freq", "sampler", "_logs_")

    def __init__(self, source_text=None, word_to_freq=None):
        """Takes text or a pregenerated histogram as input."""
        self.source_text = source_text
        word_to_freq = (word_to_freq
                        if word_to_freq else self.as_word_to_freq())
        super().__init__(word_to_freq)
        self.sampler = Sample(self)

    def as_word_to_freq(self):
        word_to_freq = defaultdict(int)

        for line in self.source_text:
            for word in self.line_as_words(line):
                sword = "".join(word)
                word_to_freq[sword] += 1
                new_freq = word_to_freq[sword]
        return word_to_freq

    def frequency(self, word):
        return self[word]

    def sample(self):
        return self.sampler.randword()


class Listogram(Gram, metaclass=LogMethodCalls, logs_size=4):
    """Listogram is a histogram implemented as a subclass of the list type."""

    __slots__ = ("temp_word_to_freq", "sampler", "_logs_")

    def __init__(self, word_list=None):
        """Initialize this histogram as a new list and count given words."""

        # hold a temporary array as new (word,counts) get added.
        # add these to new Listogram object when add_count method calls are
        # finished.
        self.temp_word_to_freq = []

        if word_list is not None:
            super().__init__(tuple(Counter(word_list).items()))
        else:
            super().__init__({})
        self.sampler = Sample(self)

    def add_count(self, word, count=1):
        """Increase frequency count of given word by given count amount."""
        # build temp array; duplicates are handled later
        self.temp_word_to_freq.append((word, count))

    def frequency(self, word):
        """Return frequency count of given word, or 0 if word is not found."""
        self.rebuild_with_latent_wordcounts()
        return self.get(word, 0)

    def __contains__(self, word):
        """Return boolean indicating if given word is in this histogram."""
        self.rebuild_with_latent_wordcounts()
        return super().__contains__(word)

    def _index(self, target):
        """Return the index of entry containing given target word if found in
        this histogram, or None if target word is not found."""
        self.rebuild_with_latent_wordcounts()
        return self.index_of(target)

    def index_of(self, target):
        """Return the index of entry containing given target word if found in
        this histogram, or None if target word is not found."""
        self.rebuild_with_latent_wordcounts()
        idx = bin_search(self.word_to_freq, target, key=lambda x: x[0])
        if idx > -1:
            return idx

    def sample(self):
        """Return a word from this histogram, randomly sampled by weighting
        each word's probability of being chosen by its observed frequency."""
        self.rebuild_with_latent_wordcounts()
        return self.sampler.randword()

    def rebuild_with_latent_wordcounts(self):
        """Reconstructs object if last method called was add_count."""
        if len(self._logs_) > 2 and self._logs_[-1 - 2].name == "add_count":
            # if the most recent class or instance method call wasn't add_count,
            # re-initialize current object.
            super().__init__(
                merge_data_containing_ints(self.word_to_freq,
                                           self.temp_word_to_freq))
            self.sampler = Sample(self)


class Dictogram(Gram, metaclass=LogMethodCalls, logs_size=4
               ):  #FreqDist, metaclass=LogMethodCalls, logs_size=4):
    """This histogram uses dict as it's primary datastructure.

    Args:


    """

    __slots__ = ("temp_word_to_freq", "word_list", "sampler")

    def __init__(self, word_list=None):

        # Temporarily hold the (word,count) added from add_count, which will
        # be added to a new distribution as part of a new Dictogram
        self.temp_word_to_freq = defaultdict(int)

        if word_list is not None:
            super().__init__(Counter(word_list))
        else:
            super().__init__({})
        self.sampler = Sample(self)

    def items(self):
        self.rebuild_with_latent_wordcounts()
        return self.word_to_freq.items()

    def add_count(self, word, count=1):
        """TIME EXPENSIVE: must call super().__init__() every time
        Increase frequency count of given word by given count amount."""
        self.temp_word_to_freq[word] += count

    def frequency(self, word):
        """Return frequency count of given word, or 0 if word is not found."""
        self.rebuild_with_latent_wordcounts()
        return self.get(word, 0)

    def sample(self):
        """Return a word from this histogram, randomly sampled by weighting
        each word's probability of being chosen by its observed frequency."""
        self.rebuild_with_latent_wordcounts()
        return self.sampler.randword()

    def rebuild_with_latent_wordcounts(self):
        """Reconstructs object if last method called was add_count."""
        if len(self._logs_) > 2 and self._logs_[-1 - 2].name == "add_count":
            # if the most recent class or instance method call wasn't add_count,
            # re-initialize current object.
            super().__init__(
                merge_data_containing_ints(self.word_to_freq,
                                           self.temp_word_to_freq))
            self.sampler = Sample(self)


class Fuzzygram(Gram, metaclass=LogMethodCalls, logs_size=4):
    pass


class Covergram(Gram, metaclass=LogMethodCalls, logs_size=4):
    """Takes a coverage report and generates a histogram of modules and their
    corresponding code coverage as a percent."""
    __slots__ = ("coverage", "coverage_data", "sampler", "_logs_")

    def __init__(self, filepath):
        self.coverage = Coverage(data_file=filepath)
        self.coverage.load()
        self.coverage_data = CoverageData()
        self.coverage_data.read_file(filepath)
        module_to_coverage = tuple(self.as_module_to_coverage())
        super().__init__(module_to_coverage)
        self.sampler = Sample(self)

    def as_module_to_coverage(self):
        """Give next (<module name>, <percent module code coverage>) in
        coverage file, a format that can be passed to the FreqDist constructor"""
        original_stdout = sys.stdout  # store original stdout state

        # temporarily suppress standard output since calling
        # coverage.Coverage.report causes yucky output
        sys.stdout = open(devnull, "w")
        line_counts = self.coverage_data.line_counts()
        total_lines = sum(count for count in line_counts.values())
        for module in self.coverage_data._lines:

            # assign coverage as a rational number between 0 and 100
            module_coverage = Fraction(self.coverage.report(module))

            yield module, int(module_coverage)
        sys.stdout = original_stdout  # restore standard output

    def frequency(self, module):
        return self.get(module, 0)

    def sample(self):
        raise self.sampler.rand_word()


class Fuzzygram(Gram, metaclass=LogMethodCalls, logs_size=4):
    pass
