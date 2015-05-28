# standard libraries
import unittest
import logging
# internal references
from opendeep.data.standard_datasets.image.cifar10 import CIFAR10
import opendeep.data.dataset as dataset
import opendeep.log.logger as logger


class TestCifar10(unittest.TestCase):

    def setUp(self):
        # configure the root logger
        logger.config_root_logger()
        # get a logger for this session
        self.log = logging.getLogger(__name__)
        # get the mnist dataset
        self.cifar = CIFAR10(one_hot=True)

    def testSizes(self):
        print self.cifar.length


    def tearDown(self):
        del self.cifar


if __name__ == '__main__':
    unittest.main()