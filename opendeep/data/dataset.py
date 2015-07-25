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
# internal imports
from opendeep.utils.misc import raise_to_list

log = logging.getLogger(__name__)

class Dataset(object):
    """
    Default interface for a dataset object.

    Attributes
    ----------
    train_inputs : iterable or list(iterable)
        The list of input iterables to use as data to the model for training.
    train_targets : iterable or list(iterable) or None
        The list of target iterables (labels) to use as labels to the model for training.
    valid_inputs : iterable or list(iterable) or None
        The list of input iterables to use as data to the model for validation.
    valid_targets : iterable or list(iterable) or None
        The list of target iterables (labels) to use as labels to the model for validation.
    test_inputs : iterable or list(iterable) or None
        The list of input iterables to use as data to the model for testing.
    test_targets : iterable or list(iterable) or None
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
        train_inputs : iterable or list(iterable)
            The list of input iterables to use as data to the model for training.
        train_targets : iterable or list(iterable), optional
            The list of target iterables (labels) to use as labels to the model for training.
        valid_inputs : iterable or list(iterable), optional
            The list of input iterables to use as data to the model for validation.
        valid_targets : iterable or list(iterable), optional
            The list of target iterables (labels) to use as labels to the model for validation.
        test_inputs : iterable or list(iterable), optional
            The list of input iterables to use as data to the model for testing.
        test_targets : iterable or list(iterable), optional
            The list of target iterables (labels) to use as labels to the model for testing.
        """
        self.train_inputs = _check_type_and_return_as_list(train_inputs, "train_inputs")
        self.train_targets = _check_type_and_return_as_list(train_targets, "train_targets")

        self.valid_inputs = _check_type_and_return_as_list(valid_inputs, "valid_inputs")
        self.valid_targets = _check_type_and_return_as_list(valid_targets, "valid_targets")

        self.test_inputs = _check_type_and_return_as_list(test_inputs, "test_inputs")
        self.test_targets = _check_type_and_return_as_list(test_targets, "test_targets")

def _check_type_and_return_as_list(iterables, name="Unknown"):
    """
    Helper method that checks the input to see if it is iterable as well as not a generator.
    (inputs the list of iterables as well as the name you want to use for this grouping of iterables,
    such as train_inputs, etc.)
    """
    iterables = raise_to_list(iterables)
    if iterables is not None:
        # type checking to make sure everything is iterable (and warn against generators).
        for idx, elem in enumerate(iterables):
            assert isinstance(elem, Iterable), "%s (as a list) parameter index %d is not iterable!" % (name, idx)
            assert not isinstance(elem, Generator), "%s (as a list) parameter index %d is a generator! " \
                                                    "Because we loop through the data multiple times, the generator " \
                                                    "will run out after the first iteration. Please consider using " \
                                                    "one of the stream types in opendeep.data.stream instead, " \
                                                    "or define your own class that performs the generator function " \
                                                    "in an __iter__(self) method!" % (name, idx)
        # if we only have one stream, just return it not in a list wrapper
        if len(iterables) == 1:
            iterables = iterables[0]
    return iterables
