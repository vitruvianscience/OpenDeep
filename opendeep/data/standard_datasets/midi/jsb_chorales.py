'''
Object for the JSB Chorales midi dataset.

Pre-processed from here: http://www-etud.iro.umontreal.ca/~boulanni/icml2012
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
import os
import glob
# third party imports
import numpy
import theano
# internal imports
from opendeep import dataset_shared
import opendeep.data.dataset as datasets
from opendeep.data.dataset import FileDataset
from opendeep.utils.midi import midiread

log = logging.getLogger(__name__)

class JSBChorales(FileDataset):
    '''
    Object for the JSB Chorales midi dataset. Pickled file of midi piano roll provided by Montreal's
    Nicolas Boulanger-Lewandowski into train, valid, and test sets.

    Attributes
    ----------
    train_shapes : list(tuple)
        List of the shapes for all the training sequences. List of (N, D) tuples for N elements in sequence
        and D dimensionality for each sequence.
    valid_shapes : list(tuple)
        List of the shapes for all the validation sequences.
    test_shapes : list(tuple)
        List of the shapes for all the testing sequences.
    train : shared variable
        The shared variable of all the training sequences concatenated into one matrix.
    valid : shared variable
        The shared variable of all the validation sequences concatenated into one matrix.
    test : shared variable
        The shared variable of all the testing sequences concatenated into one matrix.
    '''
    def __init__(self, dataset_dir='../../datasets'):
        """
        Parameters
        ----------
        dataset_dir : str
            The `dataset_dir` parameter to a ``FileDataset``.
        """
        filename = 'JSBChorales.zip'
        source = 'http://www-etud.iro.umontreal.ca/~boulanni/JSB%20Chorales.zip'

        super(JSBChorales, self).__init__(filename=filename, source=source, dataset_dir=dataset_dir)

        # now the file path has been installed to self.dataset_locations directory
        # grab the appropriate filenames
        train_filenames = os.path.join(self.dataset_location, 'JSB Chorales', 'train', '*.mid')
        valid_filenames = os.path.join(self.dataset_location, 'JSB Chorales', 'valid', '*.mid')
        test_filenames = os.path.join(self.dataset_location, 'JSB Chorales', 'test', '*.mid')
        # glob the files
        train_files = glob.glob(train_filenames)
        valid_files = glob.glob(valid_filenames)
        test_files = glob.glob(test_filenames)
        # grab the datasets from midireading the files
        train_datasets = [midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(theano.config.floatX) for f in train_files]
        valid_datasets = [midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(theano.config.floatX) for f in valid_files]
        test_datasets = [midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(theano.config.floatX) for f in test_files]
        # get the data shapes
        self.train_shapes = [train.shape for train in train_datasets]
        self.valid_shapes = [valid.shape for valid in valid_datasets]
        self.test_shapes = [test.shape for test in test_datasets]
        # put them into shared variables
        log.debug('Putting JSBChorales into theano shared variables')
        self.train = dataset_shared(numpy.concatenate(train_datasets), name='jsb_train', borrow=True)
        self.valid = dataset_shared(numpy.concatenate(valid_datasets), name='jsb_valid', borrow=True)
        self.test  = dataset_shared(numpy.concatenate(test_datasets), name='jsb_test', borrow=True)

    def getSubset(self, subset):
        """
        Returns the (x, None) pair of shared variables for the given train, validation, or test subset.

        Parameters
        ----------
        subset : int
            The subset indicator. Integer assigned by global variables in opendeep.data.dataset.py

        Returns
        -------
        tuple
            (x, None) tuple of shared variables holding the dataset input, or None if the subset doesn't exist.
        """
        if subset is datasets.TRAIN:
            return self.train, None
        elif subset is datasets.VALID:
            return self.valid, None
        elif subset is datasets.TEST:
            return self.test, None
        else:
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
            Return the list of shapes of this dataset's subset sequences. This will separate out the shapes for each
            sequence individually as items in the list, while the dataset is still concatenated into a single matrix.
        '''
        if subset not in [datasets.TRAIN, datasets.VALID, datasets.TEST]:
            log.error('Subset %s not recognized!', datasets.get_subset_strings(subset))
            return None
        if subset is datasets.TRAIN:
            return self.train_shapes
        elif subset is datasets.VALID:
            return self.valid_shapes
        elif subset is datasets.TEST:
            return self.test_shapes