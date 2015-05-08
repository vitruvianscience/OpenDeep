'''
Provides the MNIST handwritten digit dataset.

See: http://yann.lecun.com/exdb/mnist/
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
import gzip
# third party libraries
import numpy
# internal imports
from opendeep import dataset_shared, binarize
import opendeep.data.dataset as datasets
from opendeep.data.dataset import FileDataset
from opendeep.utils import file_ops
from opendeep.utils.misc import numpy_one_hot

try:
    import cPickle as pickle
except ImportError:
    import pickle

log = logging.getLogger(__name__)

class MNIST(FileDataset):
    '''
    Object for the MNIST handwritten digit dataset. Pickled file provided by Montreal's LISA lab into
    train, valid, and test sets. http://www.iro.umontreal.ca/~lisa/deep/data/mnist/

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
    def __init__(self, binary=False, binary_cutoff=0.5, one_hot=False, concat_train_valid=False,
                 dataset_dir='../../datasets', sequence_number=0, rng=None):
        """
        Parameters
        ----------
        binary : bool, optional
            Flag to binarize the input images.
        binary_cutoff : float, optional
            If you want to binarize the input images, what threshold value to use.
        one_hot : bool, optional
            Flag to convert the labels to one-hot encoding rather than their normal integers.
        concat_train_valid : bool, optional
            Flag to concatenate the training and validation datasets together. This would be the original split.
        dataset_dir : str, optional
            The `dataset_dir` parameter to a ``FileDataset``.
        sequence_number : int, optional
            The sequence method to use if we want to put the input images into a specific order. 0 defaults to random.
        rng : random, optional
            The random number generator to use when sequencing.
        """
        # instantiate the Dataset class to install the dataset from the url
        log.info('Loading MNIST with binary=%s and one_hot=%s', str(binary), str(one_hot))

        filename = 'mnist.pkl.gz'
        source = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

        super(MNIST, self).__init__(filename=filename, source=source, dataset_dir=dataset_dir)

        # self.dataset_location now contains the os path to the dataset file
        # self.file_type tells how to load the dataset
        # load the dataset into memory
        if self.file_type is file_ops.GZ:
            (self.train_X, self.train_Y), (self.valid_X, self.valid_Y), (self.test_X, self.test_Y) = pickle.load(
                gzip.open(self.dataset_location, 'rb')
            )
        else:
            (self.train_X, self.train_Y), (self.valid_X, self.valid_Y), (self.test_X, self.test_Y) = pickle.load(
                open(self.dataset_location, 'r')
            )

        if concat_train_valid:
            log.debug('Concatenating train and valid sets together...')
            self.train_X = numpy.concatenate((self.train_X, self.valid_X))
            self.train_Y = numpy.concatenate((self.train_Y, self.valid_Y))

        # sequence the dataset
        if sequence_number is not None:
            self.sequence(sequence_number=sequence_number, rng=rng)

        # make optional binary
        if binary:
            log.debug('Making MNIST X values binary with cutoff %s', str(binary_cutoff))
            self.train_X = binarize(self.train_X, binary_cutoff)
            self.valid_X = binarize(self.valid_X, binary_cutoff)
            self.test_X  = binarize(self.test_X, binary_cutoff)

        # make optional one-hot labels
        if one_hot:
            self.train_Y = numpy_one_hot(self.train_Y, n_classes=10)
            self.valid_Y = numpy_one_hot(self.valid_Y, n_classes=10)
            self.test_Y  = numpy_one_hot(self.test_Y, n_classes=10)

        log.debug("loading datasets into shared variables")
        self.train_X = dataset_shared(self.train_X, name='mnist_train_x', borrow=True)
        self.train_Y = dataset_shared(self.train_Y, name='mnist_train_y', borrow=True)

        self.valid_X = dataset_shared(self.valid_X, name='mnist_valid_x', borrow=True)
        self.valid_Y = dataset_shared(self.valid_Y, name='mnist_valid_y', borrow=True)

        self.test_X = dataset_shared(self.test_X, name='mnist_test_x', borrow=True)
        self.test_Y = dataset_shared(self.test_Y, name='mnist_test_y', borrow=True)

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
            return self._train_shape
        elif subset is datasets.VALID:
            return self._valid_shape
        elif subset is datasets.TEST:
            return self._test_shape
        else:
            log.error('Subset %s not recognized!', datasets.get_subset_strings(subset))
            return None

    def sequence(self, sequence_number, rng=None):
        """
        Sequences the train, valid, and test datasets according to the artificial sequences I made up...

        Parameters
        ----------
        sequence_number : {0, 1, 2, 3, 4}
            The sequence is is determined as follows:

            ======  =================================================
            value   Description
            ======  =================================================
            0       The original image ordering.
            1       Order by digits 0-9 repeating.
            2       Order by digits 0-9-9-0 repeating.
            3       Rotates digits 1, 4, and 8. See implementation.
            4       Has 3 bits of parity. See implementation.
            ======  =================================================

        rng : random
            the random number generator to use
        """
        log.debug("Sequencing MNIST with sequence %d", sequence_number)
        if rng is None:
            rng = numpy.random
            rng.seed(1)

        # Find the order of MNIST data going from 0-9 repeating if the first dataset
        train_ordered_indices = None
        valid_ordered_indices = None
        test_ordered_indices  = None
        if sequence_number == 0:
            pass
        elif sequence_number == 1:
            train_ordered_indices = _sequence1_indices(self.train_Y)
            valid_ordered_indices = _sequence1_indices(self.valid_Y)
            test_ordered_indices  = _sequence1_indices(self.test_Y)
        elif sequence_number == 2:
            train_ordered_indices = _sequence2_indices(self.train_Y)
            valid_ordered_indices = _sequence2_indices(self.valid_Y)
            test_ordered_indices  = _sequence2_indices(self.test_Y)
        elif sequence_number == 3:
            train_ordered_indices = _sequence3_indices(self.train_Y)
            valid_ordered_indices = _sequence3_indices(self.valid_Y)
            test_ordered_indices  = _sequence3_indices(self.test_Y)
        elif sequence_number == 4:
            train_ordered_indices = _sequence4_indices(self.train_Y)
            valid_ordered_indices = _sequence4_indices(self.valid_Y)
            test_ordered_indices  = _sequence4_indices(self.test_Y)
        else:
            log.warning("MNIST sequence number %s not recognized, leaving dataset as-is.", str(sequence_number))

        # Put the data sets in order
        if train_ordered_indices is not None and valid_ordered_indices is not None and test_ordered_indices is not None:
            self.train_X = self.train_X[train_ordered_indices]
            self.train_Y = self.train_Y[train_ordered_indices]
            self.valid_X = self.valid_X[valid_ordered_indices]
            self.valid_Y = self.valid_Y[valid_ordered_indices]
            self.test_X  = self.test_X[test_ordered_indices]
            self.test_Y  = self.test_Y[test_ordered_indices]

        # re-set the sizes
        self._train_shape = self.train_X.shape
        self._valid_shape = self.valid_X.shape
        self._test_shape = self.test_X.shape
        log.debug('Train shape is: %s', str(self._train_shape))
        log.debug('Valid shape is: %s', str(self._valid_shape))
        log.debug('Test shape is: %s', str(self._test_shape))

def _sequence1_indices(labels, classes=10):
    # make sure labels are integers
    labels = [label.astype('int32') for label in labels]
    # Creates an ordering of indices for this MNIST label series (normally expressed as y in dataset)
    # that makes the numbers go in order 0-9....
    sequence = []
    pool = []
    for _ in range(classes):
        pool.append([])
    # organize the indices into groups by label
    for i in range(len(labels)):
        pool[labels[i]].append(i)
    # draw from each pool (also with the random number insertions) until one is empty
    stop = False
    # check if there is an empty class
    for n in pool:
        if len(n) == 0:
            stop = True
            log.warning("stopped early from dataset1 sequencing - missing some class of labels")
    while not stop:
        # for i in range(classes)+range(classes-2,0,-1):
        for i in range(classes):
            if not stop:
                if len(pool[i]) == 0:  # stop the procedure if you are trying to pop from an empty list
                    stop = True
                else:
                    sequence.append(pool[i].pop())
    return sequence

# order sequentially up then down 0-9-9-0....
def _sequence2_indices(labels, classes=10):
    # make sure labels are integers
    labels = [label.astype('int32') for label in labels]
    sequence = []
    pool = []
    for _ in range(classes):
        pool.append([])
    # organize the indices into groups by label
    for i in range(len(labels)):
        pool[labels[i]].append(i)
    # draw from each pool (also with the random number insertions) until one is empty
    stop = False
    # check if there is an empty class
    for n in pool:
        if len(n) == 0:
            stop = True
            log.warning("stopped early from dataset2a sequencing - missing some class of labels")
    while not stop:
        for i in range(classes)+range(classes-1,-1,-1):
            if not stop:
                if len(pool[i]) == 0:  # stop the procedure if you are trying to pop from an empty list
                    stop = True
                else:
                    sequence.append(pool[i].pop())
    return sequence

def _sequence3_indices(labels, classes=10):
    # make sure labels are integers
    labels = [label.astype('int32') for label in labels]
    sequence = []
    pool = []
    for _ in range(classes):
        pool.append([])
    # organize the indices into groups by label
    for i in range(len(labels)):
        pool[labels[i]].append(i)
    # draw from each pool (also with the random number insertions) until one is empty
    stop = False
    # check if there is an empty class
    for n in pool:
        if len(n) == 0:
            stop = True
            log.warning("stopped early from dataset3 sequencing - missing some class of labels")
    a = False
    while not stop:
        for i in range(classes):
            if not stop:
                n=i
                if i == 1 and a:
                    n = 4
                elif i == 4 and a:
                    n = 8
                elif i == 8 and a:
                    n = 1
                if len(pool[n]) == 0:  # stop the procedure if you are trying to pop from an empty list
                    stop = True
                else:
                    sequence.append(pool[n].pop())
        a = not a

    return sequence

# extra bits of parity
def _sequence4_indices(labels, classes=10):
    # make sure labels are integers
    labels = [label.astype('int32') for label in labels]
    def even(n):
        return n % 2 == 0
    def odd(n):
        return not even(n)
    sequence = []
    pool = []
    for _ in range(classes):
        pool.append([])
    # organize the indices into groups by label
    for i in range(len(labels)):
        pool[labels[i]].append(i)
    # draw from each pool (also with the random number insertions) until one is empty
    stop = False
    # check if there is an empty class
    for n in pool:
        if len(n) == 0:
            stop = True
            log.warning("stopped early from dataset4 sequencing - missing some class of labels")
    s = [0, 1, 2]
    sequence.append(pool[0].pop())
    sequence.append(pool[1].pop())
    sequence.append(pool[2].pop())
    while not stop:
        if odd(s[-3]):
            first_bit = (s[-2] - s[-3]) % classes
        else:
            first_bit = (s[-2] + s[-3]) % classes
        if odd(first_bit):
            second_bit = (s[-1] - first_bit) % classes
        else:
            second_bit = (s[-1] + first_bit) % classes
        if odd(second_bit):
            next_num = (s[-1] - second_bit) % classes
        else:
            next_num = (s[-1] + second_bit + 1) % classes

        if len(pool[next_num]) == 0:  # stop the procedure if you are trying to pop from an empty list
            stop = True
        else:
            s.append(next_num)
            sequence.append(pool[next_num].pop())

    return sequence