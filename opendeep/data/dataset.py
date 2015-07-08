"""
Generic structure for a dataset. This defines iterable objects (streams) for data and labels to use
with any given subsets of the dataset.

Attributes
----------
TRAIN : int
    The integer representing the training dataset subset.
VALID : int
    The integer representing the validation dataset subset.
TEST : int
    The integer representing the testing dataset subset.

.. todo:: Add large dataset support with database connections, numpy.memmap, h5py, pytables
    (and in the future grabbing from pipelines like spark)

.. todo:: Add methods for cleaning data, like normalizing to mean0 std1, or scaling to [min,max].
"""
# TODO: add large dataset support with database connections, numpy.memmap, h5py, pytables
# TODO: (and in the future grabbing from pipelines like spark)
# TODO: Add methods for cleaning data, like normalizing to mean0 std1, or scaling to [min,max].

# standard libraries
import logging
from collections import Iterable
# internal imports
from opendeep.utils.misc import raise_to_list

log = logging.getLogger(__name__)

# variables for each subset of the dataset
TRAIN = 0
VALID = 1
TEST  = 2
_subsets = {TRAIN: "TRAIN", VALID: "VALID", TEST: "TEST"}

def get_subset_strings(subset):
    """
    Converts the subset integer to a string representation (e.g. TRAIN, VALID, TEST).

    Parameters
    ----------
    subset : int
        The integer specifying the subset.

    Returns
    -------
    str
        The string representation of the subset.
    """
    return _subsets.get(subset, str(subset))

class Dataset(object):
    """
    Default interface for a dataset object. At minimum, a Dataset needs to implement get_subset().
    get_subset() returns the (data, label) pair of iterables over the specific subset of this dataset.

    Attributes
    ----------
    train_inputs : list(iterable)
        The list of input iterables to use as data to the model for training.
    train_targets : list(iterable) or None
        The list of target iterables (labels) to use as labels to the model for training.
    valid_inputs : list(iterable) or None
        The list of input iterables to use as data to the model for validation.
    valid_targets : list(iterable) or None
        The list of target iterables (labels) to use as labels to the model for validation.
    test_inputs : list(iterable) or None
        The list of input iterables to use as data to the model for testing.
    test_targets : list(iterable) or None
        The list of target iterables (labels) to use as labels to the model for testing.
    """
    def __init__(self, train_inputs, train_targets=None,
                 valid_inputs=None, valid_targets=None,
                 test_inputs=None, test_targets=None):
        """
        Initialize a Dataset object that holds training, validation, and testing iterables for data
        and targets.

        Parameters
        ----------
        train_inputs : list(iterable)
            The list of input iterables to use as data to the model for training.
        train_targets : list(iterable), optional
            The list of target iterables (labels) to use as labels to the model for training.
        valid_inputs : list(iterable), optional
            The list of input iterables to use as data to the model for validation.
        valid_targets : list(iterable), optional
            The list of target iterables (labels) to use as labels to the model for validation.
        test_inputs : list(iterable), optional
            The list of input iterables to use as data to the model for testing.
        test_targets : list(iterable), optional
            The list of target iterables (labels) to use as labels to the model for testing.
        """
        self.train_inputs = raise_to_list(train_inputs)
        self.train_targets = raise_to_list(train_targets)

        self.valid_inputs = raise_to_list(valid_inputs)
        self.valid_targets = raise_to_list(valid_targets)

        self.test_inputs = raise_to_list(test_inputs)
        self.test_targets = raise_to_list(test_targets)

        # type checking to make sure everything is iterable.
        for idx, elem in enumerate(self.train_inputs):
            assert isinstance(elem, Iterable), "train_inputs parameter index %d is not iterable!" % idx
        if self.train_targets:
            for idx, elem in enumerate(self.train_targets):
                assert isinstance(elem, Iterable), "train_targets parameter index %d is not iterable!" % idx

        if self.valid_inputs:
            for idx, elem in enumerate(self.valid_inputs):
                assert isinstance(elem, Iterable), "valid_inputs parameter index %d is not iterable!" % idx
        if self.valid_targets:
            for idx, elem in enumerate(self.valid_targets):
                assert isinstance(elem, Iterable), "valid_targets parameter index %d is not iterable!" % idx

        if self.test_inputs:
            for idx, elem in enumerate(self.test_inputs):
                assert isinstance(elem, Iterable), "test_inputs parameter index %d is not iterable!" % idx
        if self.test_targets:
            for idx, elem in enumerate(self.test_targets):
                assert isinstance(elem, Iterable), "test_targets parameter index %d is not iterable!" % idx
