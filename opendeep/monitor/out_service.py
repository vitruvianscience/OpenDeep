"""
This module is for interacting with outputs for :class:`Monitor` objects (i.e. send the output to a file or database).
"""

__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
import os
# third party
from theano.compat.six import string_types  # for basestring compatability
# internal
from opendeep.utils.file_ops import mkdir_p
from opendeep.data.dataset import TRAIN, VALID, TEST

log = logging.getLogger(__name__)


class OutService(object):
    """
    Basic template for an OutService - needs a write() method to send the value to its proper destination.
    """
    def write(self, value, subset):
        """
        Given a value and the train/valid/test subset, send the value to the appropriate location.

        Parameters
        ----------
        value : object
            The value to write in the service.
        subset : int
            The subset that the value was created from (integers for data subsets determined in the
            opendeep.data.dataset module as attributes `TRAIN` `VALID` and `TEST`)

        Raises
        ------
        NotImplementedError
            If the method hasn't been implemented for the class yet.
        """
        log.exception("write() not implemented for %s!" % str(type(self)))
        raise NotImplementedError("write() not implemented for %s!" % str(type(self)))


class FileService(OutService):
    """
    Defines an OutService to write output to a given file.

    Attributes
    ----------
    train_filename : str
        Location for the file to write outputs from training.
    valid_filename : str
        Location for the file to write outputs from validation.
    test_filename : str
        Location for the file to write outputs from testing.
    """
    def __init__(self, filename):
        """
        Initialize a FileService and create empty train, valid, and test files from the given base filename.

        Parameters
        ----------
        filename : str
            Base filepath to use for the train, valid, and test files.
        """
        assert isinstance(filename, string_types), "input filename needs to be a string, found %s" % str(type(filename))
        self.value_separator = os.linesep
        filename = os.path.realpath(filename)
        basedir = os.path.dirname(filename)
        mkdir_p(basedir)
        # create the appropriate train, valid, test versions of the file
        name = os.path.basename(filename)
        name, ext = os.path.splitext(name)
        self.train_filename = os.path.join(basedir, name+'_train'+ext)
        self.valid_filename = os.path.join(basedir, name+'_valid'+ext)
        self.test_filename  = os.path.join(basedir, name+'_test'+ext)
        # init the files to be empty
        with open(self.train_filename, 'wb') as f:
            f.write('')
        with open(self.valid_filename, 'wb') as f:
            f.write('')
        with open(self.test_filename, 'wb') as f:
            f.write('')

    def write(self, value, subset):
        """
        Given a value and the train/valid/test subset indicator, append the value to the appropriate file.

        Parameters
        ----------
        value : object
            The value to append to the file.
        subset : int
            The subset that the value was created from (integers for data subsets determined in the
            opendeep.data.dataset module as attributes `TRAIN` `VALID` and `TEST`)
        """
        val_to_write = str(value) + self.value_separator
        if subset == TRAIN:
            with open(self.train_filename, 'ab') as f:
                f.write(val_to_write)
        elif subset == VALID:
            with open(self.valid_filename, 'ab') as f:
                f.write(val_to_write)
        elif subset == TEST:
            with open(self.test_filename, 'ab') as f:
                f.write(val_to_write)