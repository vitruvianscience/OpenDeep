"""
Generic structure for a dataset.

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
    get_subset() returns the (data, label) pair of iterators over the specific subset of this dataset with the
    given batch size.
    """
    def get_subset(self, subset, batch_size=1, min_batch_size=1):
        """
        This method returns the single tuple of (input_data, labels) iterators over the given subset with the given
        batch size. If an iterator doesn't exist (input_data and/or labels) for the subset, returns None in place of
        the iterator.

        Parameters
        ----------
        subset : int
            The subset indicator. Integer assigned by :mod:`opendeep.data.dataset`'s attributes.
        batch_size : int
            The number of examples to include for each iteration.
        batch_size : int
            The minimum number of examples to include for each iteration.

        Returns
        -------
        tuple
            (x, y) tuple of iterators over the dataset inputs and labels. If there aren't any labels (it is
            and unsupervised dataset), it should return (x, None).
            If the subset doesn't exist, it should return (None, None)
        """
        log.critical('No get_subset method implemented for %s!', str(type(self)))
        raise NotImplementedError()
