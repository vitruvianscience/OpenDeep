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
from opendeep.data.dataset import Dataset

log = logging.getLogger(__name__)

class NumpyDataset(Dataset):
    """
    Dataset object wrapper for something given in memory (numpy matrix, theano matrix). You pass the array-like objects
    containing the subset inputs and labels.
    """
    def __init__(self, train_inputs, train_targets=None,
                 valid_inputs=None, valid_targets=None,
                 test_inputs=None, test_targets=None,
                 train_split=1., valid_split=0.):
        """
        Initialize training, validation, and testing data from memory.

        Parameters
        ----------
        train_inputs : array
            The training input data.
        train_targets : array, optional
            The training target (label) data.
        valid_inputs : array
            The validation input data.
        valid_targets : array, optional
            The validation target (label) data.
        test_inputs : array
            The testing input data.
        test_targets : array, optional
            The testing target (label) data.
        train_split : float, optional
            The percentage of data to be used for training. This is only used if valid and test inputs are None.
        valid_split : float, optional
            The percentage of data to be used for validation. This is only used if valid and test inputs are None.
            (leftover percentage from train and valid splits will be for testing).
        """
        log.info('Wrapping dataset from memory object')

        # make sure the inputs are arrays
        train_inputs = _raise_to_array(train_inputs)
        train_targets = _raise_to_array(train_targets)

        valid_inputs = _raise_to_array(valid_inputs)
        valid_targets = _raise_to_array(valid_targets)

        test_inputs = _raise_to_array(test_inputs)
        test_targets = _raise_to_array(test_targets)

        # if validation and test sets were None, use the given split.
        if all([valid_inputs is None, valid_targets is None, test_inputs is None, test_targets is None]):
            assert (0. < train_split <= 1.), \
                "Train_split needs to be a fraction between (0, 1]. Was %f" % train_split
            assert (0. <= valid_split < 1.), \
                "Valid_split needs to be a fraction between [0, 1). Was %f" % valid_split
            assert train_split + valid_split <= 1., \
                "Train_split + valid_split can't be greater than 1. Was %f" % (train_split+valid_split)
            # make test_split the leftover percentage!
            test_split = 1 - (train_split + valid_split)

            # split up train_X and train_Y into validation and test as well
            length = train_inputs.shape[0]
            train_len = int(math.floor(length * train_split))
            valid_len = int(math.floor(length * valid_split))
            test_len = int(math.floor(length * test_split))

            # do the splits!
            if valid_len > 0:
                valid_inputs = train_inputs[train_len:train_len + valid_len]
                if train_targets is not None:
                    valid_targets = train_targets[train_len:train_len + valid_len]

            if test_len > 0:
                test_inputs = train_inputs[train_len + valid_len:]
                if train_targets is not None:
                    test_targets = train_targets[train_len + valid_len:]

            train_inputs = train_inputs[:train_len]
            if train_targets is not None:
                train_targets = train_targets[:train_len]

        super(NumpyDataset, self).__init__(train_inputs, train_targets,
                                           valid_inputs, valid_targets,
                                           test_inputs, test_targets)

def _raise_to_array(input):
    """
    Helper method to return a numpy array of the input while preserving None (because numpy.asarray() makes its own
    value for None).
    """
    if input is not None:
        return numpy.asarray(input)
    else:
        return None
