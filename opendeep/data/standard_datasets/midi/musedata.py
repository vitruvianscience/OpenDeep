'''
.. module:: musedata

Object for the MuseData midi dataset
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
from opendeep import sharedX
import opendeep.data.dataset as datasets
from opendeep.data.dataset import FileDataset
from opendeep.utils.midi import midiread

log = logging.getLogger(__name__)

class MuseData(FileDataset):
    '''
    Object for the MuseData midi dataset. Pickled file of midi piano roll provided by Montreal's
    Nicolas Boulanger-Lewandowski into train, valid, and test sets.
    '''
    def __init__(self, dataset_dir='../../datasets'):
        log.debug("Loading MuseData midi dataset...")

        filename = 'MuseData.zip'
        source = 'http://www-etud.iro.umontreal.ca/~boulanni/MuseData.zip'

        super(MuseData, self).__init__(filename=filename, source=source, dataset_dir=dataset_dir)

        # now the file path has been installed to self.dataset_locations directory
        # grab the appropriate filenames
        train_filenames = os.path.join(self.dataset_location, 'MuseData', 'train', '*.mid')
        valid_filenames = os.path.join(self.dataset_location, 'MuseData', 'valid', '*.mid')
        test_filenames  = os.path.join(self.dataset_location, 'MuseData', 'test', '*.mid')
        # glob the files
        train_files = glob.glob(train_filenames)
        valid_files = glob.glob(valid_filenames)
        test_files  = glob.glob(test_filenames)
        # grab the datasets from midireading the files
        train_datasets = [midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(theano.config.floatX) for f in train_files]
        valid_datasets = [midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(theano.config.floatX) for f in valid_files]
        test_datasets  = [midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(theano.config.floatX) for f in test_files]
        # get the data shapes
        self.train_shapes = [train.shape for train in train_datasets]
        self.valid_shapes = [valid.shape for valid in valid_datasets]
        self.test_shapes  = [test.shape for test in test_datasets]
        # put them into shared variables
        log.debug('Putting MuseData into theano shared variables')
        # self.train = sharedX(numpy.concatenate(train_datasets), borrow=True)
        # self.valid = sharedX(numpy.concatenate(valid_datasets), borrow=True)
        # self.test  = sharedX(numpy.concatenate(test_datasets), borrow=True)
        self.train = numpy.concatenate(train_datasets)
        self.valid = numpy.concatenate(valid_datasets)
        self.test = numpy.concatenate(test_datasets)

    def getDataByIndices(self, indices, subset):
        if subset not in [datasets.TRAIN, datasets.VALID, datasets.TEST]:
            log.error('Subset %s not recognized!', datasets.get_subset_strings(subset))
            return None
        if subset is datasets.TRAIN:
            return self.train[indices]#.eval()
        elif subset is datasets.VALID:
            return self.valid[indices]#.eval()
        elif subset is datasets.TEST:
            return self.test[indices]#.eval()

    def getLabelsByIndices(self, indices, subset):
        # no labels
        return None

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