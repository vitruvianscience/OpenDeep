"""
.. module:: out_service

This module is for interacting with outputs for Monitor objects (i.e. send the output to a file or database)
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
    def __init__(self):
        pass
    def write(self, value, subset):
        pass

class FileService(OutService):
    def __init__(self, filename):
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