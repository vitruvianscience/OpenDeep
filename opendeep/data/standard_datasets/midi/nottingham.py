"""
Object for the Nottingham midi dataset.

Pre-processed from here: http://www-etud.iro.umontreal.ca/~boulanni/icml2012
"""
# standard libraries
import logging
# third party
import numpy
from theano import config
# internal imports
from opendeep.utils.file_ops import find_files
from opendeep.data.dataset_file import FileDataset
from opendeep.utils.midi import midiread
from opendeep.utils.decorators import inherit_docs

log = logging.getLogger(__name__)

@inherit_docs
class Nottingham(FileDataset):
    """
    Object for the Nottingham midi dataset. Pickled file of midi piano roll provided by Montreal's
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
    def __init__(self, path='datasets/Nottingham',
                 source='http://www-etud.iro.umontreal.ca/~boulanni/Nottingham.zip',
                 train_filter='.*train.*',
                 valid_filter='.*valid.*',
                 test_filter='.*test.*', ):

        super(Nottingham, self).__init__(path=path, source=source,
                                       train_filter=train_filter,
                                       valid_filter=valid_filter,
                                       test_filter=test_filter)

        # grab the datasets from midireading the files
        train_datasets = [
            midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(config.floatX)
            for f in find_files(self.path, train_filter)
            ]
        valid_datasets = [
            midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(config.floatX)
            for f in find_files(self.path, valid_filter)
            ]
        test_datasets = [
            midiread(f, r=(21, 109), dt=0.3).piano_roll.astype(config.floatX)
            for f in find_files(self.path, test_filter)
            ]

        self.train_inputs = numpy.concatenate(train_datasets)
        self.train_targets = None

        self.valid_inputs = numpy.concatenate(valid_datasets)
        self.valid_targets = None

        self.test_inputs = numpy.concatenate(test_datasets)
        self.test_targets = None
