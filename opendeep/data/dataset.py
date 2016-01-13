"""
Generic structure for a dataset. This defines iterable objects (streams) for data and labels to use
with any given subsets of the dataset.

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
from types import GeneratorType as Generator

_log = logging.getLogger(__name__)

class Dataset(object):
    """
    Default interface for a dataset object.

    Attributes
    ----------
    train_inputs : iterable
        The input iterable to use as data to the model for training.
    train_targets : iterable or None
        The target iterable (labels) to use as labels to the model for training.
    valid_inputs : iterable or None
        The input iterable to use as data to the model for validation.
    valid_targets : iterable or None
        The target iterable (labels) to use as labels to the model for validation.
    test_inputs : iterable or None
        The input iterable to use as data to the model for testing.
    test_targets : iterable or None
        The target iterable (labels) to use as labels to the model for testing.
    """
    def __init__(self, train_inputs, train_targets=None,
                 valid_inputs=None, valid_targets=None,
                 test_inputs=None, test_targets=None):
        """
        Initialize a Dataset object that holds training, validation, and testing iterables for data
        and targets.

        Parameters
        ----------
        train_inputs : iterable
            The iterable to use as data to the model for training.
        train_targets : iterable, optional
            The target iterable (labels) to use as labels to the model for training.
        valid_inputs : iterable, optional
            The input iterable to use as data to the model for validation.
        valid_targets : iterable, optional
            The target iterable (labels) to use as labels to the model for validation.
        test_inputs : iterable, optional
            The input iterable to use as data to the model for testing.
        test_targets : iterable, optional
            The target iterable (labels) to use as labels to the model for testing.
        """
        self.train_inputs = _check_type(train_inputs, "train_inputs")
        self.train_targets = _check_type(train_targets, "train_targets")

        self.valid_inputs = _check_type(valid_inputs, "valid_inputs")
        self.valid_targets = _check_type(valid_targets, "valid_targets")

        self.test_inputs = _check_type(test_inputs, "test_inputs")
        self.test_targets = _check_type(test_targets, "test_targets")

def _check_type(iterable, name="Unknown"):
    """
    Helper method that checks the input to see if it is iterable as well as not a generator.
    (inputs the iterable as well as the name you want to use for this iterable,
    such as train_inputs, etc.)
    """
    if iterable is not None:
        # type checking to make sure everything is iterable (and warn against generators).
        assert isinstance(iterable, Iterable), "%s is not iterable! Found %s" % \
                                               (name, str(type(iterable)))
        assert not isinstance(iterable, Generator), "%s is a generator! " \
                                                    "Because we loop through the data multiple times, the generator " \
                                                    "will run out after the first iteration. Please consider using " \
                                                    "one of the stream types in opendeep.data.stream instead, " \
                                                    "or define your own class that performs the generator function " \
                                                    "in an __iter__(self) method!" % name
    return iterable
