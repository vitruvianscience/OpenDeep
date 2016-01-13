# standard libraries
import unittest
import logging
# third party
import numpy
# internal references
from opendeep.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def setUp(self):
        # get a logger for this session
        self.log = logging.getLogger(__name__)
        # create dataset
        self.train = [[1, 2], [4, 5]]
        self.trainl = [2, 3]

        self.valid = numpy.array([[2, 3], [5, 6], [8, 9]])
        self.validl = numpy.array([3, 6, 9])

        self.test = numpy.array([[3, 4], [6, 7], [1, 2], [9, 0]])
        self.testl = numpy.array([4, 7, 2, 0])

        self.dataset = Dataset(train_inputs=self.train, train_targets=self.trainl,
                               valid_inputs=self.valid, valid_targets=self.validl,
                               test_inputs=self.test, test_targets=self.testl)

    def testTrain(self):
        assert numpy.array_equal(self.dataset.train_inputs, self.train)
        assert numpy.array_equal(self.dataset.train_targets, self.trainl)

    def testValid(self):
        assert numpy.array_equal(self.dataset.valid_inputs, self.valid)
        assert numpy.array_equal(self.dataset.valid_targets, self.validl)

    def testTest(self):
        assert numpy.array_equal(self.dataset.test_inputs, self.test)
        assert numpy.array_equal(self.dataset.test_targets, self.testl)

    def testGeneratorInput(self):
        gen = (x for x in [1, 2, 3, 4])
        ran = True
        try:
            _ = Dataset(train_inputs=gen)
        except AssertionError:
            ran = False
        except Exception as e:
            raise e

        assert ran is False, "The bad dataset (with a generator as input) succesfully initialized."

    def testNonIterableInput(self):
        ran = True
        try:
            _ = Dataset(train_inputs=100)
        except AssertionError:
            ran = False
        except Exception as e:
            raise e

        assert ran is False, "The bad dataset (with a non-iterable [int] as input) succesfully initialized."

    def tearDown(self):
        del self.dataset


if __name__ == '__main__':
    unittest.main()
