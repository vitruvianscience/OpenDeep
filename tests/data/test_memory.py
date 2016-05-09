# standard libraries
import unittest
# third party
import numpy
# internal references
from opendeep.data.dataset_memory import NumpyDataset

class TestMemoryDataset(unittest.TestCase):

    def setUp(self):
        # create dataset
        self.train = numpy.array([[1, 2], [4, 5]])
        self.valid = numpy.array([[2, 3], [5, 6], [8, 9]])
        self.test  = numpy.array([[3, 4], [6, 7], [1, 2], [9, 0]])
        self.dataset = NumpyDataset(train_inputs=self.train, valid_inputs=self.valid, test_inputs=self.test)

    def testElements(self):
        for elem, actual in zip(self.dataset.train_inputs, self.train):
            assert numpy.array_equal(elem, actual)

        for elem, actual in zip(self.dataset.valid_inputs, self.valid):
            assert numpy.array_equal(elem, actual)

        for elem, actual in zip(self.dataset.test_inputs, self.test):
            assert numpy.array_equal(elem, actual)

    def tearDown(self):
        del self.dataset, self.train, self.valid, self.test


if __name__ == '__main__':
    unittest.main()
