"""
Provides the MNIST handwritten digit dataset.

See: http://yann.lecun.com/exdb/mnist/
"""
# standard libraries
import logging
import gzip
import math
# third party libraries
import numpy
# internal imports
from opendeep.data.dataset_file import FileDataset
from opendeep.utils import file_ops
from opendeep.utils.misc import numpy_one_hot, binarize

try:
    import cPickle as pickle
except ImportError:
    import pickle

log = logging.getLogger(__name__)

# python 2 vs. python 3 mnist source
import sys
if sys.version_info > (3, 0):
    mnist_source = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist_py3k.pkl.gz'
    mnist_path = 'datasets/mnist_py3k.pkl.gz'
else:
    mnist_source = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    mnist_path = 'datasets/mnist.pkl.gz'

class MNIST(FileDataset):
    """
    Object for the MNIST handwritten digit dataset. Pickled file provided by Montreal's LISA lab into
    train, valid, and test sets. http://www.iro.umontreal.ca/~lisa/deep/data/mnist/
    """
    def __init__(self, binary=False, binary_cutoff=0.5, one_hot=False, concat_train_valid=False,
                 sequence_number=0, seq_3d=False, seq_length=30, rng=None,
                 path=mnist_path,
                 source=mnist_source):
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
        sequence_number : int, optional
            The sequence method to use if we want to put the input images into a specific order. 0 defaults to random.
        seq_3d : bool, optional
            When sequencing, whether the output should be
            3D tensors (batches, subsequences, data) or 2D (sequence, data).
        rng : random, optional
            The random number generator to use when sequencing.
        path : str, optional
            The `path` parameter to a ``FileDataset``.
        source : str, optional
            The `source` parameter to a ``FileDataset``.
        """
        # instantiate the Dataset class to install the dataset from the url
        log.info('Loading MNIST with binary=%s and one_hot=%s', str(binary), str(one_hot))

        super(MNIST, self).__init__(path=path, source=source)

        # self.path now contains the os path to the dataset file
        # self.file_type tells how to load the dataset
        # load the dataset into memory
        if self.file_type is file_ops.GZ:
            (self.train_inputs, self.train_targets), \
            (self.valid_inputs, self.valid_targets), \
            (self.test_inputs, self.test_targets) = pickle.load(
                gzip.open(self.path, 'rb')
            )
        else:
            (self.train_inputs, self.train_targets), \
            (self.valid_inputs, self.valid_targets), \
            (self.test_inputs, self.test_targets) = pickle.load(
                open(self.path, 'r')
            )

        if concat_train_valid:
            log.debug('Concatenating train and valid sets together...')
            self.train_inputs = numpy.concatenate((self.train_inputs, self.valid_inputs))
            self.train_targets = numpy.concatenate((self.train_targets, self.valid_targets))

        # sequence the dataset
        if sequence_number is not None:
            self._sequence(sequence_number=sequence_number, rng=rng)

        # make optional binary
        if binary:
            log.debug('Making MNIST X values binary with cutoff %s', str(binary_cutoff))
            self.train_inputs = binarize(self.train_inputs, binary_cutoff)
            self.valid_inputs = binarize(self.valid_inputs, binary_cutoff)
            self.test_inputs  = binarize(self.test_inputs, binary_cutoff)

        # make optional one-hot labels
        if one_hot:
            self.train_targets = numpy_one_hot(self.train_targets, n_classes=10)
            self.valid_targets = numpy_one_hot(self.valid_targets, n_classes=10)
            self.test_targets  = numpy_one_hot(self.test_targets, n_classes=10)

        # optionally make 3D instead of 2D
        if seq_3d:
            log.debug("Making 3D....")
            # chop up into sequences of length seq_length
            # first make sure to chop off the remainder of the data so seq_length can divide evenly.
            if self.train_inputs.shape[0] % seq_length != 0:
                length, dim = self.train_inputs.shape
                if self.train_targets.ndim == 1:
                    ydim = 1
                else:
                    ydim = self.train_targets.shape[-1]
                self.train_inputs = self.train_inputs[:seq_length * math.floor(length / seq_length)]
                self.train_targets = self.train_targets[:seq_length * math.floor(length / seq_length)]
                # now create the 3D tensor of sequences - they will be (num_sequences, sequence_size, 784)
                self.train_inputs = numpy.reshape(self.train_inputs, (length / seq_length, seq_length, dim))
                self.train_targets = numpy.reshape(self.train_targets, (length / seq_length, seq_length, ydim))

            if self.valid_inputs.shape[0] % seq_length != 0:
                length, dim = self.valid_inputs.shape
                if self.valid_targets.ndim == 1:
                    ydim = 1
                else:
                    ydim = self.valid_targets.shape[-1]
                self.valid_inputs = self.valid_inputs[:seq_length * math.floor(length / seq_length)]
                self.valid_targets = self.valid_targets[:seq_length * math.floor(length / seq_length)]
                # now create the 3D tensor of sequences - they will be (num_sequences, sequence_size, 784)
                self.valid_inputs = numpy.reshape(self.valid_inputs, (length / seq_length, seq_length, dim))
                self.valid_targets = numpy.reshape(self.valid_targets, (length / seq_length, seq_length, ydim))

            if self.test_inputs.shape[0] % seq_length != 0:
                length, dim = self.test_inputs.shape
                if self.test_targets.ndim == 1:
                    ydim = 1
                else:
                    ydim = self.test_targets.shape[-1]
                self.test_inputs = self.test_inputs[:seq_length * math.floor(length / seq_length)]
                self.test_targets = self.test_targets[:seq_length * math.floor(length / seq_length)]
                # now create the 3D tensor of sequences - they will be (num_sequences, sequence_size, 784)
                self.test_inputs = numpy.reshape(self.test_inputs, (length / seq_length, seq_length, dim))
                self.test_targets = numpy.reshape(self.test_targets, (length / seq_length, seq_length, ydim))

            self._train_shape = self.train_inputs.shape
            self._valid_shape = self.valid_inputs.shape
            self._test_shape = self.test_inputs.shape
            log.debug('Train shape is: %s', str(self._train_shape))
            log.debug('Valid shape is: %s', str(self._valid_shape))
            log.debug('Test shape is: %s', str(self._test_shape))

    def _sequence(self, sequence_number, rng=None):
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
            train_ordered_indices = _sequence1_indices(self.train_targets)
            valid_ordered_indices = _sequence1_indices(self.valid_targets)
            test_ordered_indices  = _sequence1_indices(self.test_targets)
        elif sequence_number == 2:
            train_ordered_indices = _sequence2_indices(self.train_targets)
            valid_ordered_indices = _sequence2_indices(self.valid_targets)
            test_ordered_indices  = _sequence2_indices(self.test_targets)
        elif sequence_number == 3:
            train_ordered_indices = _sequence3_indices(self.train_targets)
            valid_ordered_indices = _sequence3_indices(self.valid_targets)
            test_ordered_indices  = _sequence3_indices(self.test_targets)
        elif sequence_number == 4:
            train_ordered_indices = _sequence4_indices(self.train_targets)
            valid_ordered_indices = _sequence4_indices(self.valid_targets)
            test_ordered_indices  = _sequence4_indices(self.test_targets)
        else:
            log.warning("MNIST sequence number %s not recognized, leaving dataset as-is.", str(sequence_number))

        # Put the data sets in order
        if train_ordered_indices is not None and valid_ordered_indices is not None and test_ordered_indices is not None:
            self.train_inputs = self.train_inputs[train_ordered_indices]
            self.train_targets = self.train_targets[train_ordered_indices]
            self.valid_inputs = self.valid_inputs[valid_ordered_indices]
            self.valid_targets = self.valid_targets[valid_ordered_indices]
            self.test_inputs  = self.test_inputs[test_ordered_indices]
            self.test_targets  = self.test_targets[test_ordered_indices]

        # re-set the sizes
        self._train_shape = self.train_inputs.shape
        self._valid_shape = self.valid_inputs.shape
        self._test_shape = self.test_inputs.shape
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
