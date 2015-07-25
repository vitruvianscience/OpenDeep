# standard libraries
import unittest
import logging
# internal references
from opendeep.data.standard_datasets.image.mnist import MNIST
from opendeep.log.logger import config_root_logger

class TestMNIST(unittest.TestCase):

    def setUp(self):
        # configure the root logger
        config_root_logger()
        # get a logger for this session
        self.log = logging.getLogger(__name__)
        # get the mnist dataset
        self.mnist = MNIST(path="../../../datasets/mnist.pkl.gz", binary=False, concat_train_valid=True)

    def testSizes(self):
        pass

    def tearDown(self):
        del self.mnist


if __name__ == '__main__':
    unittest.main()