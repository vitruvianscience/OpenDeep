"""
Generic structure for a dataset wrapper around array-like memory objects (like numpy array).

.. todo:: Add a class for Pandas dataframes.
"""
# TODO: add a class for Pandas dataframes.
# standard libraries
import logging
import math
# third party libraries
import numpy
# internal imports
from opendeep.data.dataset import Dataset, TRAIN, VALID, TEST, _subsets
from opendeep.utils.decorators import inherit_docs

log = logging.getLogger(__name__)

@inherit_docs
class NumpyDataset(Dataset):
    """
    Dataset object wrapper for something given in memory (numpy matrix, theano matrix). You pass the array-like objects
    containing the subset inputs and labels.
    """
    def __init__(self, train_x, train_y=None, valid_x=None, valid_y=None, test_x=None, test_y=None,
                 train_split=1., valid_split=0.):
        """
        Initialize training, validation, and testing data from memory.

        Parameters
        ----------
        train_x : array
            The training input data.
        train_y : array, optional
            The training target (label) data.
        valid_x : array
            The validation input data.
        valid_y : array, optional
            The validation target (label) data.
        test_x : array
            The testing input data.
        test_y : array, optional
            The testing target (label) data.
        train_split : float, optional
            The percentage of data to be used for training. This is only used if valid and test inputs are None.
        valid_split : float, optional
            The percentage of data to be used for validation. This is only used if valid and test inputs are None.
            (leftover percentage from train and valid splits will be for testing).
        """
        log.info('Wrapping dataset from memory object')

        # make sure the inputs are arrays
        train_x = numpy.asarray(train_x)
        if train_y is not None:
            train_y = numpy.asarray(train_y)

        if valid_x is not None:
            valid_x = numpy.asarray(valid_x)
        if valid_y is not None:
            valid_y = numpy.asarray(valid_y)

        if test_x is not None:
            test_x = numpy.asarray(test_x)
        if test_y is not None:
            test_y = numpy.asarray(test_y)

        # if validation and test sets were None, use the given split.
        if all([valid_x is None, valid_y is None, test_x is None, test_y is None]):
            assert (0. < train_split <= 1.), \
                "Train_split needs to be a fraction between (0, 1]. Was %f" % train_split
            assert (0. <= valid_split < 1.), \
                "Valid_split needs to be a fraction between [0, 1). Was %f" % valid_split
            assert train_split + valid_split <= 1., \
                "Train_split + valid_split can't be greater than 1. Was %f" % (train_split+valid_split)
            # make test_split the leftover percentage!
            test_split = 1 - (train_split + valid_split)

            # split up train_X and train_Y into validation and test as well
            length = train_x.shape[0]
            train_len = int(math.floor(length * train_split))
            valid_len = int(math.floor(length * valid_split))
            test_len = int(math.floor(length * test_split))

            # do the splits!
            if valid_len > 0:
                valid_x = train_x[train_len:train_len + valid_len]
                if train_y is not None:
                    valid_y = train_y[train_len:train_len + valid_len]

            if test_len > 0:
                test_x = train_x[train_len + valid_len:]
                if train_y is not None:
                    test_y = train_y[train_len + valid_len:]

            train_x = train_x[:train_len]
            if train_y is not None:
                train_y = train_y[:train_len]

        self.datasets = {TRAIN: [train_x, train_y], VALID: [valid_x, valid_y], TEST: [test_x, test_y]}

    def get_subset(self, subset):
        # make sure the subset is valid.
        assert subset in _subsets, "Subset %s not recognized!" % str(subset)

        # return the appropriately sized iterators over the subset.
        # in this case, we are returning the numpy arrays.
        return self.datasets[subset]
