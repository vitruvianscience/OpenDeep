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
            return self.train_X.get_value(borrow=True)[indices]
        elif subset is datasets.VALID and hasattr(self, 'valid_X') and self.valid_X:
            return self.valid_X.get_value(borrow=True)[indices]
        elif subset is datasets.TEST and hasattr(self, 'test_X') and self.test_X:
            return self.test_X.get_value(borrow=True)[indices]
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
            return self.train_Y.get_value(borrow=True)[indices]
        elif subset is datasets.VALID and hasattr(self, 'valid_Y') and self.valid_Y:
            return self.valid_Y.get_value(borrow=True)[indices]
        elif subset is datasets.TEST and hasattr(self, 'test_Y') and self.test_Y:
            return self.test_Y.get_value(borrow=True)[indices]
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