"""
Cifar-10 standard image dataset.
"""
# standard libraries
import logging
import os
import math
# third party libraries
import numpy
# internal imports
from opendeep.data.dataset import TRAIN, VALID, TEST, get_subset_strings
from opendeep.data.dataset_file import FileDataset
from opendeep.utils.misc import numpy_one_hot
from opendeep.utils.decorators import inherit_docs

try:
    import cPickle as pickle
except ImportError:
    import pickle

log = logging.getLogger(__name__)

@inherit_docs
class CIFAR10(FileDataset):
    """
    Object for the CIFAR-10 image dataset.
    The CIFAR-10 dataset consists of 60000 32x32 color images in 10 classes, with 6000 images per class.
    There are 50000 training images and 10000 test images.

    This dataset object only considers the 50000 training images and creates splits from there.

    http://www.cs.toronto.edu/~kriz/cifar.html

    Attributes
    ----------
    train_X : shared variable
        The training input variables.
    train_Y : shared variable
        The training input labels.
    valid_X : shared variable
        The validation input variables.
    valid_Y : shared variable
        The validation input labels.
    test_X : shared variable
        The testing input variables.
    test_Y : shared variable
        The testing input labels.
    """
    def __init__(self, train_split=0.95, valid_split=0.05, one_hot=False,
                 path='../../datasets/cifar-10-batches-py/',
                 source='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'):
        """
        Parameters
        ----------
        train_split : float
            The percentage of data to be used for training.
        valid_split : float
            The percentage of data to be used for validation.
            (leftover percentage from train and valid splits will be for testing).
        one_hot : bool, optional
            Flag to convert the labels to one-hot encoding rather than their normal integers.
        path : str, optional
            The `path` parameter to a ``FileDataset``.
        source : str, optional
            The `source` parameter to a ``FileDataset``.
        """
        assert (0. < train_split <= 1.), "Train_split needs to be a fraction between (0, 1]."
        assert (0. <= valid_split < 1.), "Valid_split needs to be a fraction between [0, 1)."
        assert train_split + valid_split <= 1., "Train_split + valid_split can't be greater than 1."
        # make test_split the leftover percentage!
        test_split = 1 - (train_split + valid_split)

        # instantiate the Dataset class to install the dataset from the url
        log.info('Loading CIFAR-10 with data split (%f, %f, %f)' %
                 (train_split, valid_split, test_split))

        super(CIFAR10, self).__init__(path=path, source=source)

        # extract out all the samples
        # (from keras https://github.com/fchollet/keras/blob/master/keras/datasets/cifar10.py)
        nb_samples = 50000
        X = numpy.zeros((nb_samples, 3, 32, 32), dtype="uint8")
        Y = numpy.zeros((nb_samples,), dtype="uint8")
        for i in range(1, 6):
            fpath = os.path.join(self.path, 'data_batch_%d' % i)
            with open(fpath, 'rb') as f:
                d = pickle.load(f)
            data = d['data']
            labels = d['labels']

            data = data.reshape(data.shape[0], 3, 32, 32)
            X[(i - 1) * 10000:i * 10000, :, :, :] = data
            Y[(i - 1) * 10000:i * 10000] = labels

        if one_hot:
            Y = numpy_one_hot(Y, n_classes=10)

        self.length = X.shape[0]

        self._train_len = int(math.floor(self.length * train_split))
        self._valid_len = int(math.floor(self.length * valid_split))
        self._test_len = int(max(self.length - self._valid_len - self._train_len, 0))

        # divide into train, valid, and test sets!
        self.train_X = X[:self._train_len]
        self.train_Y = Y[:self._train_len]

        if valid_split > 0:
            self.valid_X = X[self._train_len:self._train_len + self._valid_len]
            self.valid_Y = Y[self._train_len:self._train_len + self._valid_len]
        else:
            self.valid_X = None
            self.valid_Y = None

        if test_split > 0:
            self.test_X = X[self._train_len + self._valid_len:]
            self.test_Y = Y[self._train_len + self._valid_len:]
        else:
            self.test_X = None
            self.test_Y = None

    def get_subset(self, subset):
        """
        Returns the (x, y) pair of shared variables for the given train, validation, or test subset.

        Parameters
        ----------
        subset : int
            The subset indicator. Integer assigned by global variables in opendeep.data.dataset.py

        Returns
        -------
        tuple
            (x, y) tuple of shared variables holding the dataset input and label, or None if the subset doesn't exist.
        """
        if subset is TRAIN:
            return self.train_X, self.train_Y
        elif subset is VALID:
            return self.valid_X, self.valid_Y
        elif subset is TEST:
            return self.test_X, self.test_Y
        else:
            log.error('Subset %s not recognized!', get_subset_strings(subset))
            return None, None
