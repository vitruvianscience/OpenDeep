from __future__ import division, absolute_import, print_function

from .dataset import TRAIN, VALID, TEST

# to get dataset types like Dataset, Filedataset, etc.
from .dataset import Dataset
from .dataset_file import FileDataset
from .dataset_memory import NumpyDataset
# to get premade datasets
from .standard_datasets import *