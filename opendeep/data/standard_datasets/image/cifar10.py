__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
import os
import math
# third party libraries
import numpy
# internal imports
from opendeep import dataset_shared
import opendeep.data.dataset as datasets
from opendeep.data.dataset import FileDataset
from opendeep.utils import file_ops
from opendeep.utils.misc import numpy_one_hot

try:
    import cPickle as pickle
except ImportError:
    import pickle

log = logging.getLogger(__name__)

class CIFAR10(FileDataset):
    '''
    Object for the CIFAR-10 image dataset.
    The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
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
    '''
    def __init__(self, train_split=0.95, valid_split=0.05, one_hot=False,
                 dataset_dir='../../datasets'):
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
        dataset_dir : str, optional
            The `dataset_dir` parameter to a ``FileDataset``.
        """
        filename = 'cifar-10-python.tar.gz'
        source = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        unzipped_dir = 'cifar-10-batches-py'

        assert (0. < train_split <= 1.), "Train_split needs to be a fraction between (0, 1]."
        assert (0. <= valid_split < 1.), "Valid_split needs to be a fraction between [0, 1)."
        assert train_split + valid_split <= 1., "Train_split + valid_split can't be greater than 1."
        # make test_split the leftover percentage!
        test_split = 1 - (train_split + valid_split)

        # instantiate the Dataset class to install the dataset from the url
        log.info('Loading CIFAR-10 with data split (%f, %f, %f)' %
                 (train_split, valid_split, test_split))

        super(CIFAR10, self).__init__(filename=filename, source=source, dataset_dir=dataset_dir)

        # self.dataset_location now contains the os path to the dataset file
        # self.file_type tells how to load the dataset
        # load the dataset into memory
        # extract the tarball if necessary (if the cifar-10-batches-py directory doesn't exist).
        unzipped_loc = os.path.join(os.path.dirname(self.dataset_location), unzipped_dir)
        if not os.path.exists(unzipped_loc):
            # make sure it is a tarball
            if self.file_type is file_ops.GZ:
                    file_ops.untar(self.dataset_location, os.path.dirname(self.dataset_location))
            else:
                raise AssertionError("Didn't find a .gz file! The file should be %s" % filename)

        # extract out all the samples
        # (from keras https://github.com/fchollet/keras/blob/master/keras/datasets/cifar10.py)
        nb_samples = 50000
        X = numpy.zeros((nb_samples, 3, 32, 32), dtype="uint8")
        Y = numpy.zeros((nb_samples,), dtype="uint8")
        for i in range(1, 6):
            fpath = os.path.join(unzipped_loc, 'data_batch_%d' % i)
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
        log.debug("loading datasets into shared variables")
        self.train_X = dataset_shared(X[:self._train_len], name='cifar10_train_x', borrow=True)
        self.train_Y = dataset_shared(Y[:self._train_len], name='cifar10_train_y', borrow=True)

        if valid_split > 0:
            self.valid_X = dataset_shared(X[self._train_len:self._train_len + self._valid_len],
                                          name='cifar10_valid_x', borrow=True)
            self.valid_Y = dataset_shared(Y[self._train_len:self._train_len + self._valid_len],
                                          name='cifar10_valid_y', borrow=True)
        else:
            self.valid_X = None
            self.valid_Y = None

        if test_split > 0:
            self.test_X = dataset_shared(X[self._train_len + self._valid_len:],
                                         name='cifar10_test_x', borrow=True)
            self.test_Y = dataset_shared(Y[self._train_len + self._valid_len:],
                                         name='cifar10_test_y', borrow=True)
        else:
            self.test_X = None
            self.test_Y = None

    def getSubset(self, subset):
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
        if subset is datasets.TRAIN:
            return self.train_X, self.train_Y
        elif subset is datasets.VALID:
            return self.valid_X, self.valid_Y
        elif subset is datasets.TEST:
            return self.test_X, self.test_Y
        else:
            log.error('Subset %s not recognized!', datasets.get_subset_strings(subset))
            return None, None

    def getDataShape(self, subset):
        '''
        Returns the shape of the input data for the given subset

        Parameters
        ----------
        subset : int
            The subset indicator. Integer assigned by global variables in opendeep.data.dataset.py

        Returns
        -------
        tuple
            Return the shape of this dataset's subset in a (N, D) tuple where N=#examples and D=dimensionality
        '''
        if subset is datasets.TRAIN:
            return self._train_len, 3, 32, 32
        elif subset is datasets.VALID:
            return self._valid_len, 3, 32, 32
        elif subset is datasets.TEST:
            return self._test_len, 3, 32, 32
        else:
            log.error('Subset %s not recognized!', datasets.get_subset_strings(subset))
            return None