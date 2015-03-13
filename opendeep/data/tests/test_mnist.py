'''
Unit testing for the mnist dataset
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import unittest
import logging
# internal references
from opendeep.data.standard_datasets.image.mnist import MNIST
import opendeep.data.dataset as dataset
import opendeep.log.logger as logger
from opendeep.data.iterators.sequential import SequentialIterator
from opendeep.data.iterators.random import RandomIterator


class TestMNIST(unittest.TestCase):

    def setUp(self):
        # configure the root logger
        logger.config_root_logger()
        # get a logger for this session
        self.log = logging.getLogger(__name__)
        # get the mnist dataset
        self.mnist = MNIST()
        # instantiate the sequential iterator
        self.sequentialIterator = SequentialIterator(self.mnist, dataset.TRAIN, 255, 255)
        # instantiate the random iterator
        self.randomIterator = RandomIterator(self.mnist, dataset.TRAIN, 255, 255)

    def testSizes(self):
        assert self.mnist.getDataShape(dataset.TRAIN) == (60000, 784)
        assert self.mnist.getDataShape(dataset.VALID) == (10000, 784)
        assert self.mnist.getDataShape(dataset.TEST) == (10000, 784)

    def testSequentialIterator(self):
        self.log.debug('TESTING SEQUENTIAL ITERATOR')
        i = 0
        for _, y in self.sequentialIterator:
            if i < 2:
                self.log.debug(y)
            i+=1
        assert i==235

    def testRandomIterator(self):
        self.log.debug('TESTING RANDOM ITERATOR')
        i = 0
        for x, y in self.randomIterator:
            if i < 2:
                self.log.debug(y)
            i+=1
        assert i==235

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()