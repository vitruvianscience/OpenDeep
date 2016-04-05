# standard libraries
import unittest
import logging
# internal references
from opendeep.data.standard_datasets.image.cifar10 import CIFAR10
from opendeep.log.logger import config_root_logger


class TestCifar10(unittest.TestCase):

    def setUp(self):
        # configure the root logger
        config_root_logger()
        # get a logger for this session
        self.log = logging.getLogger(__name__)
        # get the mnist dataset
        # self.cifar = CIFAR10(one_hot=True, path='../../../datasets/cifar-10-batches-py/')
        self.cifar = CIFAR10(one_hot=True)

    def testSizes(self):
        print("lengths:")
        print("\ntrain")
        if self.cifar.train_inputs is not None:
            print(len(self.cifar.train_inputs))
        if self.cifar.train_targets is not None:
            print(len(self.cifar.train_targets))

        print("\nvalid")
        if self.cifar.valid_inputs is not None:
            print(len(self.cifar.valid_inputs))
        if self.cifar.valid_targets is not None:
            print(len(self.cifar.valid_targets))

        print("\ntest")
        if self.cifar.test_inputs is not None:
            print(len(self.cifar.test_inputs))
        if self.cifar.test_targets is not None:
            print(len(self.cifar.test_targets))

    def testShapes(self):
        print("shapes:")
        print("\ninput")
        if self.cifar.train_inputs is not None:
            for i,x in enumerate(self.cifar.train_inputs):
                if i>0:
                    break
                print x.shape
        print("\ntarget")
        if self.cifar.train_targets is not None:
            for i,x in enumerate(self.cifar.train_targets):
                if i>0:
                    break
                print x
                print x.shape


    def tearDown(self):
        del self.cifar


if __name__ == '__main__':
    unittest.main()
