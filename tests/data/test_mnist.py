# standard libraries
import unittest
# internal references
from opendeep.data.standard_datasets.image.mnist import MNIST
# from opendeep.log.logger import config_root_logger


class TestMNIST(unittest.TestCase):

    def setUp(self):
        pass

    def testDefaultSizes(self):
        mnist = MNIST(path="../../datasets/mnist.pkl.gz")
        self.assertEquals(mnist.train_inputs.shape, (50000, 1, 28, 28))
        self.assertEquals(mnist.train_targets.shape, (50000,))

        self.assertEquals(mnist.valid_inputs.shape, (10000, 1, 28, 28))
        self.assertEquals(mnist.valid_targets.shape, (10000,))

        self.assertEquals(mnist.test_inputs.shape, (10000, 1, 28, 28))
        self.assertEquals(mnist.test_targets.shape, (10000,))
        del mnist

    def testFlattenSizes(self):
        mnist = MNIST(path="../../datasets/mnist.pkl.gz", concat_train_valid=False, flatten=True)
        self.assertEquals(mnist.train_inputs.shape, (50000, 784))
        self.assertEquals(mnist.train_targets.shape, (50000,))

        self.assertEquals(mnist.valid_inputs.shape, (10000, 784))
        self.assertEquals(mnist.valid_targets.shape, (10000,))

        self.assertEquals(mnist.test_inputs.shape, (10000, 784))
        self.assertEquals(mnist.test_targets.shape, (10000,))
        del mnist

    def testConcatSizes(self):
        mnist = MNIST(path="../../datasets/mnist.pkl.gz", concat_train_valid=True, flatten=False)
        self.assertEquals(mnist.train_inputs.shape, (60000, 1, 28, 28))
        self.assertEquals(mnist.train_targets.shape, (60000,))

        self.assertEquals(mnist.valid_inputs.shape, (10000, 1, 28, 28))
        self.assertEquals(mnist.valid_targets.shape, (10000,))

        self.assertEquals(mnist.test_inputs.shape, (10000, 1, 28, 28))
        self.assertEquals(mnist.test_targets.shape, (10000,))
        del mnist

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
