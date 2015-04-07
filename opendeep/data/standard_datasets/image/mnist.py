'''
.. module:: mnist

Object for the MNIST handwritten digit dataset
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
import cPickle
import gzip
# third party libraries
import numpy
# internal imports
from opendeep import make_shared_variables
import opendeep.data.dataset as datasets
from opendeep.data.dataset import FileDataset
from opendeep.utils import file_ops
from opendeep.utils.misc import numpy_one_hot

log = logging.getLogger(__name__)

class MNIST(FileDataset):
    '''
    Object for the MNIST handwritten digit dataset. Pickled file provided by Montreal's LISA lab into
    train, valid, and test sets.
    '''
    def __init__(self, binary=False, one_hot=False, dataset_dir='../../datasets'):
        # instantiate the Dataset class to install the dataset from the url
        log.info('Loading MNIST with binary=%s and one_hot=%s', str(binary), str(one_hot))

        filename = 'mnist.pkl.gz'
        source = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

        super(MNIST, self).__init__(filename=filename, source=source, dataset_dir=dataset_dir)

        # self.dataset_location now contains the os path to the dataset file
        # self.file_type tells how to load the dataset
        # load the dataset into memory
        if self.file_type is file_ops.GZ:
            (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = cPickle.load(
                gzip.open(self.dataset_location, 'rb')
            )
        else:
            (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = cPickle.load(
                open(self.dataset_location, 'r')
            )

        # make optional binary
        if binary:
            _binary_cutoff = 0.5
            log.debug('Making MNIST X values binary with cutoff %s', str(_binary_cutoff))
            train_X = (train_X > _binary_cutoff).astype('float32')
            valid_X = (valid_X > _binary_cutoff).astype('float32')
            test_X  = (test_X > _binary_cutoff).astype('float32')

        # make optional one-hot labels
        if one_hot:
            train_Y = numpy_one_hot(train_Y, n_classes=10)
            valid_Y = numpy_one_hot(valid_Y, n_classes=10)
            test_Y  = numpy_one_hot(test_Y, n_classes=10)

        log.debug('Concatenating train and valid sets together...')
        train_X = numpy.concatenate((train_X, valid_X))
        train_Y = numpy.concatenate((train_Y, valid_Y))

        self._train_shape = train_X.shape
        self._valid_shape = valid_X.shape
        self._test_shape  = test_X.shape
        log.debug('Train shape is: %s', str(self._train_shape))
        log.debug('Valid shape is: %s', str(self._valid_shape))
        log.debug('Test shape is: %s', str(self._test_shape))
        # transfer the datasets into theano shared variables
        log.debug('Loading MNIST into theano shared variables')
        (self.train_X, self.train_Y,
         self.valid_X, self.valid_Y,
         self.test_X, self.test_Y) = make_shared_variables((train_X, train_Y, valid_X, valid_Y, test_X, test_Y),
                                                           borrow=True)


    def getDataByIndices(self, indices, subset):
        '''
        This method is used by an iterator to return data values at given indices.
        :param indices: either integer or list of integers
        The index (or indices) of values to return
        :param subset: integer
        The integer representing the subset of the data to consider dataset.(TRAIN, VALID, or TEST)
        :return: array
        The dataset values at the index (indices)
        '''
        if subset is datasets.TRAIN:
            # return self.train_X.get_value(borrow=True)[indices]
            return self.train_X[indices].eval()
        elif subset is datasets.VALID and hasattr(self, 'valid_X') and self.valid_X:
            return self.valid_X[indices].eval()
        elif subset is datasets.TEST and hasattr(self, 'test_X') and self.test_X:
            return self.test_X[indices].eval()
        else:
            return None

    def getLabelsByIndices(self, indices, subset):
        '''
        This method is used by an iterator to return data label values at given indices.
        :param indices: either integer or list of integers
        The index (or indices) of values to return
        :param subset: integer
        The integer representing the subset of the data to consider dataset.(TRAIN, VALID, or TEST)
        :return: array
        The dataset labels at the index (indices)
        '''
        if subset is datasets.TRAIN and hasattr(self, 'train_Y') and self.train_Y:
            return self.train_Y[indices].eval()
        elif subset is datasets.VALID and hasattr(self, 'valid_Y') and self.valid_Y:
            return self.valid_Y[indices].eval()
        elif subset is datasets.TEST and hasattr(self, 'test_Y') and self.test_Y:
            return self.test_Y[indices].eval()
        else:
            return None

    def hasSubset(self, subset):
        '''
        :param subset: integer
        The integer representing the subset of the data to consider dataset.(TRAIN, VALID, or TEST)
        :return: boolean
        Whether or not this dataset has the given subset split
        '''
        if subset not in [datasets.TRAIN, datasets.VALID, datasets.TEST]:
            log.error('Subset %s not recognized!', datasets.get_subset_strings(subset))
            return False
        # it has them all.
        return True

    def getDataShape(self, subset):
        '''
        :return: tuple
        Return the shape of this dataset's subset in a NxD tuple where N=#examples and D=dimensionality
        '''
        if subset not in [datasets.TRAIN, datasets.VALID, datasets.TEST]:
            log.error('Subset %s not recognized!', datasets.get_subset_strings(subset))
            return None
        if subset is datasets.TRAIN:
            return self._train_shape
        elif subset is datasets.VALID:
            return self._valid_shape
        elif subset is datasets.TEST:
            return self._test_shape
        else:
            log.critical('No getDataShape method implemented for %s for subset %s!',
                         str(type(self)),
                         datasets.get_subset_strings(subset))
            raise NotImplementedError()

    def sequence(self, sequence_number, rng=None, one_hot=False):
        """
        Sequences the train, valid, and test datasets according to the artificial sequences

        :param sequence_number: which sequence to do
        :type sequence_number: int

        :param rng: the random number generator to use
        :type rng: rng

        :param one_hot: whether to encode the data as one-hot
        :type one_hot: bool
        """
        log.debug("Sequencing MNIST with sequence %d", sequence_number)
        if rng is None:
            rng = numpy.random
            rng.seed(1)

        def set_xy_indices(x, y, indices):
            x.set_value(x.get_value(borrow=True)[indices])
            y.set_value(y.get_value(borrow=True)[indices])

        # Find the order of MNIST data going from 0-9 repeating if the first dataset
        train_ordered_indices = None
        valid_ordered_indices = None
        test_ordered_indices  = None
        if sequence_number == 1:
            train_ordered_indices = sequence1_indices(self.train_Y.get_value(borrow=True))
            valid_ordered_indices = sequence1_indices(self.valid_Y.get_value(borrow=True))
            test_ordered_indices  = sequence1_indices(self.test_Y.get_value(borrow=True))
        elif sequence_number == 2:
            train_ordered_indices = sequence2_indices(self.train_Y.get_value(borrow=True))
            valid_ordered_indices = sequence2_indices(self.valid_Y.get_value(borrow=True))
            test_ordered_indices  = sequence2_indices(self.test_Y.get_value(borrow=True))
        elif sequence_number == 3:
            train_ordered_indices = sequence3_indices(self.train_Y.get_value(borrow=True))
            valid_ordered_indices = sequence3_indices(self.valid_Y.get_value(borrow=True))
            test_ordered_indices  = sequence3_indices(self.test_Y.get_value(borrow=True))
        elif sequence_number == 4:
            train_ordered_indices = sequence4_indices(self.train_Y.get_value(borrow=True))
            valid_ordered_indices = sequence4_indices(self.valid_Y.get_value(borrow=True))
            test_ordered_indices  = sequence4_indices(self.test_Y.get_value(borrow=True))

        # Put the data sets in order
        if train_ordered_indices and valid_ordered_indices and test_ordered_indices:
            set_xy_indices(self.train_X, self.train_Y, train_ordered_indices)
            set_xy_indices(self.valid_X, self.valid_Y, valid_ordered_indices)
            set_xy_indices(self.test_X, self.test_Y, test_ordered_indices)


        ###############################################################################################################
        # For testing one-hot encoding to see if it can learn sequences without having to worry about nonlinear input #
        ###############################################################################################################
        if one_hot:
            # construct the numpy matrix of representations from y
            train = numpy.array([[1 if i == y else 0 for i in range(10)] for y in self.train_Y.get_value(borrow=True)],
                                dtype="float32")
            self.train_X.set_value(train)
            valid = numpy.array([[1 if i == y else 0 for i in range(10)] for y in self.valid_Y.get_value(borrow=True)],
                                dtype="float32")
            self.valid_X.set_value(valid)
            test = numpy.array([[1 if i == y else 0 for i in range(10)] for y in self.test_Y.get_value(borrow=True)],
                               dtype="float32")
            self.test_X.set_value(test)

        # re-set the sizes
        self._train_shape = self.train_X.shape.eval()
        self._valid_shape = self.valid_X.shape.eval()
        self._test_shape = self.test_X.shape.eval()
        log.debug('Train shape is: %s', str(self._train_shape))
        log.debug('Valid shape is: %s', str(self._valid_shape))
        log.debug('Test shape is: %s', str(self._test_shape))

def sequence1_indices(labels, classes=10):
    # make sure labels are integers
    labels = [label.astype('int32') for label in labels]
    #Creates an ordering of indices for this MNIST label series (normally expressed as y in dataset) that makes the numbers go in order 0-9....
    sequence = []
    pool = []
    for _ in range(classes):
        pool.append([])
    #organize the indices into groups by label
    for i in range(len(labels)):
        pool[labels[i]].append(i)
    #draw from each pool (also with the random number insertions) until one is empty
    stop = False
    #check if there is an empty class
    for n in pool:
        if len(n) == 0:
            stop = True
            log.warning("stopped early from dataset1 sequencing - missing some class of labels")
    while not stop:
        #for i in range(classes)+range(classes-2,0,-1):
        for i in range(classes):
            if not stop:
                if len(pool[i]) == 0: #stop the procedure if you are trying to pop from an empty list
                    stop = True
                else:
                    sequence.append(pool[i].pop())
    return sequence

#order sequentially up then down 0-9-9-0....
def sequence2_indices(labels, classes=10):
    # make sure labels are integers
    labels = [label.astype('int32') for label in labels]
    sequence = []
    pool = []
    for _ in range(classes):
        pool.append([])
    #organize the indices into groups by label
    for i in range(len(labels)):
        pool[labels[i]].append(i)
    #draw from each pool (also with the random number insertions) until one is empty
    stop = False
    #check if there is an empty class
    for n in pool:
        if len(n) == 0:
            stop = True
            log.warning("stopped early from dataset2a sequencing - missing some class of labels")
    while not stop:
        for i in range(classes)+range(classes-1,-1,-1):
            if not stop:
                if len(pool[i]) == 0: #stop the procedure if you are trying to pop from an empty list
                    stop = True
                else:
                    sequence.append(pool[i].pop())
    return sequence

def sequence3_indices(labels, classes=10):
    # make sure labels are integers
    labels = [label.astype('int32') for label in labels]
    sequence = []
    pool = []
    for _ in range(classes):
        pool.append([])
    #organize the indices into groups by label
    for i in range(len(labels)):
        pool[labels[i]].append(i)
    #draw from each pool (also with the random number insertions) until one is empty
    stop = False
    #check if there is an empty class
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
                if len(pool[n]) == 0: #stop the procedure if you are trying to pop from an empty list
                    stop = True
                else:
                    sequence.append(pool[n].pop())
        a = not a

    return sequence

# extra bits of parity
def sequence4_indices(labels, classes=10):
    # make sure labels are integers
    labels = [label.astype('int32') for label in labels]
    def even(n):
        return n%2==0
    def odd(n):
        return not even(n)
    sequence = []
    pool = []
    for _ in range(classes):
        pool.append([])
    #organize the indices into groups by label
    for i in range(len(labels)):
        pool[labels[i]].append(i)
    #draw from each pool (also with the random number insertions) until one is empty
    stop = False
    #check if there is an empty class
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
            first_bit = (s[-2] - s[-3])%classes
        else:
            first_bit = (s[-2] + s[-3])%classes
        if odd(first_bit):
            second_bit = (s[-1] - first_bit)%classes
        else:
            second_bit = (s[-1] + first_bit)%classes
        if odd(second_bit):
            next_num = (s[-1] - second_bit)%classes
        else:
            next_num = (s[-1] + second_bit + 1)%classes

        if len(pool[next_num]) == 0: #stop the procedure if you are trying to pop from an empty list
            stop = True
        else:
            s.append(next_num)
            sequence.append(pool[next_num].pop())

    return sequence