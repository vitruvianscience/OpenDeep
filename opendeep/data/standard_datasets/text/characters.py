'''
Provides a dataset of sequences of characters from a text file.

Based on https://github.com/karpathy/char-rnn/blob/master/util/CharSplitLMMinibatchLoader.lua
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
import time
import math
# third party libraries
import numpy
# internal imports
from opendeep.utils.constructors import dataset_shared
from opendeep.data.dataset import FileDataset, TRAIN, VALID, TEST, get_subset_strings
from opendeep.utils import file_ops
from opendeep.utils.misc import numpy_one_hot, make_time_units_string

try:
    import cPickle as pickle
except ImportError:
    import pickle

log = logging.getLogger(__name__)

class CharsLM(FileDataset):
    '''
    Object for getting character sequence from a text file. This dataset is for creating a character-level
    language model. The input will be sequences of characters from the file, and the labels will be the same
    sequence but shifted one character forward (for the task of predicting the next character in the sequence).

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
    vocab_size : int
        The size of the vocab (input_size for models).
    length : int
        The total size of the dataset # sequences.
    '''
    def __init__(self, filename, seq_length=120,
                 train_split=0.95, valid_split=0.05,
                 dataset_dir='../../datasets', source=None):
        """
        Parameters
        ----------
        filename : str
            The filename for the .txt file to use.
        seq_length : int
            The maximum length of characters to consider a sequence.
        train_split : float
            The percentage of data to be used for training.
        valid_split : float
            The percentage of data to be used for validation.
            (leftover percentage from train and valid splits will be for testing).
        dataset_dir : str, optional
            The `dataset_dir` parameter to a ``FileDataset``.
        source : str, optional
            The `source` parameter to a ``FileDataset`` for downloading the file.

        Raises
        ------
        AssertionError
            If some input parameters (split percentages, etc.) don't work.
        """
        assert (0. < train_split <= 1.), "Train_split needs to be a fraction between (0, 1]."
        assert (0. <= valid_split < 1.), "Valid_split needs to be a fraction between [0, 1)."
        assert train_split + valid_split <= 1., "Train_split + valid_split can't be greater than 1."
        # make test_split the leftover percentage!
        test_split = 1 - (train_split + valid_split)

        # instantiate the Dataset class to install the dataset from the url
        log.info('Loading characters from %s with sequence length %d and data split (%f, %f, %f)' %
                 (filename, seq_length, train_split, valid_split, test_split))

        super(CharsLM, self).__init__(filename=filename, source=source, dataset_dir=dataset_dir)

        # self.dataset_location now contains the os path to the dataset file
        # self.file_type tells how to load the dataset
        # load the dataset into memory
        if self.file_type is not file_ops.TXT:
            log.error("Filetype was not .txt, found %s" % file_ops.get_filetype_string(self.file_type))

        # try to read the file
        log.debug("Reading file...")
        t = time.time()
        try:
            with open(self.dataset_location, mode='r') as f:
                chars = f.read(3000000)
        except Exception:
            log.critical("Error reading file!")
            raise
        log.debug("Reading took %s" % make_time_units_string(time.time()-t))

        log.debug("Creating vocabulary...")
        t1 = time.time()
        unique_chars = list(set(chars))
        self.vocab = {}
        for i, c in enumerate(unique_chars):
            self.vocab[c] = i
        # now vocab contains a dictionary mapping a character to a unique integer.

        log.debug("Converting characters to one-hot")
        # convert the chars into a list of ints.
        converted_chars = numpy.asarray([self.vocab[c] for c in chars])
        self.chars = numpy_one_hot(converted_chars, n_classes=numpy.amax(self.vocab.values())+1)
        self.labels = self.chars.copy()[1:]
        self.chars = self.chars[:-1]

        log.debug("Vocab and conversion took %s" % make_time_units_string(time.time()-t1))
        del chars, converted_chars

        length, self.vocab_size = self.chars.shape
        if seq_length > length or seq_length <= 0 or seq_length is None:
            log.debug("seq_length was %s while length was %d. Changing to fit length." % (str(seq_length), length))
            seq_length = length
        # now divide into sequences of seq_length
        # first make sure to chop off the remainder of the data so seq_length can divide evenly.
        if length % seq_length != 0:
            self.chars = self.chars[:seq_length * math.floor(length / seq_length)]
            self.labels = self.labels[:seq_length * math.floor(length / seq_length)]
        # now create the 3D tensor of sequences - they will be (num_sequences, sequence_size, vocab_size)
        self.chars = numpy.reshape(self.chars, (length/seq_length, seq_length, self.vocab_size))
        self.labels = numpy.reshape(self.labels, (length/seq_length, seq_length, self.vocab_size))
        # shuffle
        self.length = self.chars.shape[0]
        shuffle_order = numpy.arange(self.length)
        numpy.random.shuffle(shuffle_order)
        self.chars = self.chars[shuffle_order]
        self.labels = self.labels[shuffle_order]

        self._train_len = int(math.floor(self.length*train_split))
        self._valid_len = int(math.floor(self.length*valid_split))
        self._test_len  = int(max(self.length - self._valid_len - self._train_len, 0))
        self._seq_len = seq_length

        # divide into train, valid, and test sets!
        log.debug("loading datasets into shared variables")
        self.train_X = dataset_shared(self.chars[:self._train_len], name='chars_train_x', borrow=True)
        self.train_Y = dataset_shared(self.labels[:self._train_len], name='chars_train_y', borrow=True)

        if valid_split > 0:
            self.valid_X = dataset_shared(self.chars[self._train_len:self._train_len+self._valid_len],
                                          name='chars_valid_x', borrow=True)
            self.valid_Y = dataset_shared(self.labels[self._train_len:self._train_len+self._valid_len],
                                          name='chars_valid_y', borrow=True)
        else:
            self.valid_X = None
            self.valid_Y = None

        if test_split > 0:
            self.test_X = dataset_shared(self.chars[self._train_len+self._valid_len:],
                                         name='chars_test_x', borrow=True)
            self.test_Y = dataset_shared(self.labels[self._train_len+self._valid_len:],
                                         name='chars_test_y', borrow=True)
        else:
            self.test_X = None
            self.test_Y = None

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
        if subset is TRAIN:
            return self.train_X, self.train_Y
        elif subset is VALID:
            return self.valid_X, self.valid_Y
        elif subset is TEST:
            return self.test_X, self.test_Y
        else:
            log.error('Subset %s not recognized!', get_subset_strings(subset))
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
        if subset is TRAIN:
            return self._train_len, self._seq_len, self.vocab_size
        elif subset is VALID:
            return self._valid_len, self._seq_len, self.vocab_size
        elif subset is TEST:
            return self._test_len, self._seq_len, self.vocab_size
        else:
            log.error('Subset %s not recognized!', get_subset_strings(subset))
            return None