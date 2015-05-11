# standard libraries
import unittest
import logging
# internal references
from opendeep.data.standard_datasets.image.mnist import MNIST
import opendeep.data.dataset as dataset
import opendeep.log.logger as logger


class TestMNIST(unittest.TestCase):

    def setUp(self):
        # configure the root logger
        logger.config_root_logger()
        # get a logger for this session
        self.log = logging.getLogger(__name__)
        # get the mnist dataset
        self.mnist = MNIST(binary=False, concat_train_valid=True)

    def testSizes(self):
        assert self.mnist.getDataShape(dataset.TRAIN) == (60000, 784)
        assert self.mnist.getDataShape(dataset.VALID) == (10000, 784)
        assert self.mnist.getDataShape(dataset.TEST) == (10000, 784)


    def tearDown(self):
        del self.mnist


if __name__ == '__main__':
    unittest.main()