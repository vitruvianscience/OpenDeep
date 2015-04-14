'''
.. module:: jsb_chorales

Object for the JSB Chorales midi dataset
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
    '''
    def __init__(self, dataset_dir='../../datasets'):

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
        if subset is datasets.TRAIN:
            return self.train, None
        elif subset is datasets.VALID:
            return self.valid, None
        elif subset is datasets.TEST:
            return self.test, None
        else:
            return None, None

    def hasSubset(self, subset):
        if subset not in [datasets.TRAIN, datasets.VALID, datasets.TEST]:
            log.error('Subset %s not recognized!', datasets.get_subset_strings(subset))
        else:
            # it has train valid and test
            return True

    def getDataShape(self, subset):
        if subset not in [datasets.TRAIN, datasets.VALID, datasets.TEST]:
            log.error('Subset %s not recognized!', datasets.get_subset_strings(subset))
            return None
        if subset is datasets.TRAIN:
            return self.train_shapes
        elif subset is datasets.VALID:
            return self.valid_shapes
        elif subset is datasets.TEST:
            return self.test_shapes