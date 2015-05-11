# standard libraries
import unittest
import logging
# third party
import numpy
# internal references
from opendeep.data.dataset import MemoryDataset
import opendeep.data.dataset as dataset
import opendeep.log.logger as logger


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

    def testSizes(self):
        assert self.dataset.getDataShape(dataset.TRAIN) == (2, 2)
        assert self.dataset.getDataShape(dataset.VALID) == (3, 2)
        assert self.dataset.getDataShape(dataset.TEST)  == (4, 2)

    def tearDown(self):
        del self.dataset


if __name__ == '__main__':
    unittest.main()