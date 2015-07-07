# standard libraries
import unittest
import logging
# third party
import numpy
# internal references
from opendeep.data.dataset_memory import NumpyDataset
from opendeep.log.logger import config_root_logger


class TestMemoryDataset(unittest.TestCase):

    def setUp(self):
        # configure the root logger
        config_root_logger()
        # get a logger for this session
        self.log = logging.getLogger(__name__)
        # create dataset
        train = numpy.array([[1, 2], [4, 5]])
        valid = numpy.array([[2, 3], [5, 6], [8, 9]])
        test  = numpy.array([[3, 4], [6, 7], [1, 2], [9, 0]])
        self.dataset = NumpyDataset(train_x=train, valid_x=valid, test_x=test)

    def tearDown(self):
        del self.dataset


if __name__ == '__main__':
    unittest.main()