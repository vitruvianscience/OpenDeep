'''
Unit testing for the mnist dataset
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import unittest
import logging
# third party
import numpy
# internal references
from opendeep.data.dataset import MemoryDataset
import opendeep.data.dataset as dataset
import opendeep.log.logger as logger
from opendeep.data.iterators.sequential import SequentialIterator
from opendeep.data.iterators.random import RandomIterator


class TestMemoryDataset(unittest.TestCase):

    def setUp(self):
        # configure the root logger
        logger.config_root_logger()
        # get a logger for this session
        self.log = logging.getLogger(__name__)
        # create dataset
        train = numpy.array([[1, 2], [4, 5]])
        valid = numpy.array([[2, 3], [5, 6], [8, 9]])
        test  = numpy.array([[3, 4], [6, 7], [1, 2], [9, 0]])
        self.dataset = MemoryDataset(train_X=train, valid_X=valid, test_X=test)
        # instantiate the sequential iterator
        self.sequentialIterator = SequentialIterator(self.dataset, dataset.TRAIN)
        # instantiate the random iterator
        self.randomIterator = RandomIterator(self.dataset, dataset.TRAIN)

    def testSizes(self):
        assert self.dataset.getDataShape(dataset.TRAIN) == (2, 2)
        assert self.dataset.getDataShape(dataset.VALID) == (3, 2)
        assert self.dataset.getDataShape(dataset.TEST)  == (4, 2)

    def testSequentialIterator(self):
        self.log.debug('TESTING SEQUENTIAL ITERATOR')
        for x, y in self.sequentialIterator:
            self.log.debug(x)
            self.log.debug(y)

    def testRandomIterator(self):
        self.log.debug('TESTING RANDOM ITERATOR')
        for x, y in self.randomIterator:
            self.log.debug(x)
            self.log.debug(y)

    def tearDown(self):
        del self.dataset
        del self.sequentialIterator
        del self.randomIterator


if __name__ == '__main__':
    unittest.main()