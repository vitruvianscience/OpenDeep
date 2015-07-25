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
from opendeep.data.dataset_file import FileDataset
from opendeep.utils.misc import numpy_one_hot

try:
    import cPickle as pickle
except ImportError:
    import pickle

log = logging.getLogger(__name__)

class CIFAR10(FileDataset):
    """
    Object for the CIFAR-10 image dataset.
    The CIFAR-10 dataset consists of 60000 32x32 color images in 10 classes, with 6000 images per class.
    There are 50000 training images and 10000 test images.

    This dataset object only considers the 50000 training images and creates splits from there.

    http://www.cs.toronto.edu/~kriz/cifar.html
    """
    def __init__(self, train_split=0.95, valid_split=0.05, one_hot=False,
                 path='datasets/cifar-10-batches-py/',
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

        length = X.shape[0]

        train_len = int(math.floor(length * train_split))
        valid_len = int(math.floor(length * valid_split))

        # divide into train, valid, and test sets!
        self.train_inputs = X[:train_len]
        self.train_targets = Y[:train_len]

        if valid_split > 0:
            self.valid_inputs = X[train_len:train_len + valid_len]
            self.valid_targets = Y[train_len:train_len + valid_len]
        else:
            self.valid_inputs = None
            self.valid_targets = None

        if test_split > 0:
            self.test_inputs = X[train_len + valid_len:]
            self.test_targets = Y[train_len + valid_len:]
        else:
            self.test_inputs = None
            self.test_targets = None
