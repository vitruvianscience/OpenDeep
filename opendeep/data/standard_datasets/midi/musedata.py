"""
Object for the MuseData midi dataset.

Pre-processed from here: http://www-etud.iro.umontreal.ca/~boulanni/icml2012
"""
# standard libraries
import logging
# third party
import numpy
import theano
# internal imports
from opendeep.data.dataset import TRAIN, VALID, TEST
from opendeep.utils.file_ops import find_files
from opendeep.data.dataset_file import FileDataset
from opendeep.utils.midi import midiread
from opendeep.utils.decorators import inherit_docs

log = logging.getLogger(__name__)

@inherit_docs
class MuseData(FileDataset):
    """
    Object for the MuseData midi dataset. Pickled file of midi piano roll provided by Montreal's
    Nicolas Boulanger-Lewandowski into train, valid, and test sets.

    Attributes
    ----------
    train : numpy matrix
        All the training sequences concatenated into one matrix.
    valid : numpy matrix
        All the validation sequences concatenated into one matrix.
    test : numpy matrix
        All the testing sequences concatenated into one matrix.
    """
    def __init__(self, path='../../datasets/MuseData',
                 source='http://www-etud.iro.umontreal.ca/~boulanni/MuseData.zip',
                 train_filter='.*train.*',
                 valid_filter='.*valid.*',
                 test_filter='.*test.*', ):

        super(MuseData, self).__init__(path=path, source=source,
                                       train_filter=train_filter,
                                       valid_filter=valid_filter,
                                       test_filter=test_filter)

        # grab the datasets from midireading the files
        train_datasets = [
            midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(theano.config.floatX)
            for f in find_files(self.path, self.train_filter)
            ]
        valid_datasets = [
            midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(theano.config.floatX)
            for f in find_files(self.path, self.valid_filter)
            ]
        test_datasets = [
            midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(theano.config.floatX)
            for f in find_files(self.path, self.test_filter)
            ]

        self.train = numpy.concatenate(train_datasets)
        self.valid = numpy.concatenate(valid_datasets)
        self.test = numpy.concatenate(test_datasets)

    def get_subset(self, subset):
        if subset is TRAIN:
            return self.train, None
        elif subset is VALID:
            return self.valid, None
        elif subset is TEST:
            return self.test, None
        else:
            return None, None
