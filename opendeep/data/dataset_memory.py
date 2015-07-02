"""
Generic structure for a dataset wrapper around array-like memory objects (like numpy array).
"""
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
import math
# third party libraries
import numpy
# internal imports
from opendeep.data.iterators.memory import NumpyBatches
from opendeep.data.dataset import Dataset, TRAIN, VALID, TEST, _subsets
from opendeep.utils.decorators import inherit_docs

log = logging.getLogger(__name__)

@inherit_docs
class MemoryDataset(Dataset):
    """
    Dataset object wrapper for something given in memory (numpy matrix, theano matrix). You pass the array-like objects
    containing the subset inputs and labels.
    """
    def __init__(self, train_X, train_Y=None, valid_X=None, valid_Y=None, test_X=None, test_Y=None,
                 train_split=1., valid_split=0.):
        """
        Initialize training, validation, and testing data from memory.

        Parameters
        ----------
        train_X : array
            The training input data.
        train_Y : array, optional
            The training target (label) data.
        valid_X : array
            The validation input data.
        valid_Y : array, optional
            The validation target (label) data.
        test_X : array
            The testing input data.
        test_Y : array, optional
            The testing target (label) data.
        train_split : float, optional
            The percentage of data to be used for training. This is only used if valid and test inputs are None.
        valid_split : float, optional
            The percentage of data to be used for validation. This is only used if valid and test inputs are None.
            (leftover percentage from train and valid splits will be for testing).
        """
        log.info('Wrapping dataset from memory object')

        # make sure the inputs are arrays
        train_X = numpy.asarray(train_X)
        if train_Y is not None:
            train_Y = numpy.asarray(train_Y)

        if valid_X is not None:
            valid_X = numpy.asarray(valid_X)
        if valid_Y is not None:
            valid_Y = numpy.asarray(valid_Y)

        if test_X is not None:
            test_X = numpy.asarray(test_X)
        if test_Y is not None:
            test_Y = numpy.asarray(test_Y)

        # if validation and test sets were None, use the given split.
        if all([valid_X==None, valid_Y==None, test_X==None, test_Y==None]):
            assert (0. < train_split <= 1.), "Train_split needs to be a fraction between (0, 1]."
            assert (0. <= valid_split < 1.), "Valid_split needs to be a fraction between [0, 1)."
            assert train_split + valid_split <= 1., "Train_split + valid_split can't be greater than 1."
            # make test_split the leftover percentage!
            test_split = 1 - (train_split + valid_split)

            # split up train_X and train_Y into validation and test as well
            length = train_X.shape[0]
            _train_len = int(math.floor(length * train_split))
            _valid_len = int(math.floor(length * valid_split))
            _test_len = int(math.floor(length * test_split))

            # do the splits!
            if _valid_len > 0:
                valid_X = train_X[_train_len:_train_len + _valid_len]
                if train_Y is not None:
                    valid_Y = train_Y[_train_len:_train_len + _valid_len]

            if _test_len > 0:
                test_X = train_X[_train_len + _valid_len:]
                if train_Y is not None:
                    test_Y = train_Y[_train_len + _valid_len:]

            train_X = train_X[:_train_len]
            if train_Y is not None:
                train_Y = train_Y[:_train_len]

        self.datasets = {TRAIN: [train_X, train_Y], VALID: [valid_X, valid_Y], TEST: [test_X, test_Y]}

    def get_subset(self, subset, batch_size=1, min_batch_size=1):
        # make sure the subset is valid.
        assert subset in _subsets, "Subset %s not recognized!" % str(subset)

        # return the appropriately sized iterator over the subset
        x, y = [
            NumpyBatches(data, batch_size, min_batch_size) if data is not None else None
            for data in self.datasets[subset]
        ]
        return x, y
