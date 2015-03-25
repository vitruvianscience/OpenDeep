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
import theano
# internal imports
from opendeep import make_shared_variables
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
        test_filenames = os.path.join(self.dataset_location, 'MuseData', 'test', '*.mid')
        # glob the files
        train_files = glob.glob(train_filenames)
        valid_files = glob.glob(valid_filenames)
        test_files = glob.glob(test_filenames)
        # grab the datasets from midireading the files
        train_datasets = [midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(theano.config.floatX) for f in train_files]
        valid_datasets = [midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(theano.config.floatX) for f in valid_files]
        test_datasets = [midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(theano.config.floatX) for f in test_files]
        # put them into shared variables
        log.debug('Putting MuseData into theano shared variables')
        self.train = make_shared_variables(train_datasets, borrow=True)
        self.valid = make_shared_variables(valid_datasets, borrow=True)
        self.test = make_shared_variables(test_datasets, borrow=True)

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
        # no labels
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
        else:
            # it has train valid and test
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
            pass
        elif subset is datasets.VALID:
            pass
        elif subset is datasets.TEST:
            pass
        else:
            log.critical('No getDataShape method implemented for %s for subset %s!',
                         str(type(self)),
                         datasets.get_subset_strings(subset))
            raise NotImplementedError()