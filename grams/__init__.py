# grams.__init__
#!/usr/bin/env python3
"""A package for using histograms to interact with data."""
import os

import nltk

__all__ = []
from .grams import *

__all__ += grams.__all__
from .online import *

__all__ += online.__all__
from .stats import FreqDist, Sample

__all__ += stats.__all__
from .utils import *

__all__ += utils.__all__

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
# add the `nltk_data` folder to nltk's list of directories to find data in
nltk.data.path.append(os.path.join(os.getcwd(), "grams/nltk_data/"))
