# standard libraries
import unittest
import logging
# third party
import numpy
# internal references
from opendeep.data.dataset import Dataset
from opendeep.utils.misc import raise_to_list


class TestMemoryDataset(unittest.TestCase):

    def setUp(self):
        # get a logger for this session
        self.log = logging.getLogger(__name__)
        # create dataset
        train = [[1, 2], [4, 5]]
        trainl = [[2, 3]]

        valid = [numpy.array([[2, 3], [5, 6], [8, 9]])]
        validl = [numpy.array([[3, 4], [6, 7], [9, 0]])]

        test = numpy.array([[3, 4], [6, 7], [1, 2], [9, 0]])
        testl = numpy.array([[4, 5], [7, 8], [2, 3], [0, 1]])

        self.dataset = Dataset(train_inputs=train, train_targets=trainl,
                               valid_inputs=valid, valid_targets=validl,
                               test_inputs=test, test_targets=testl)

    def testTrain(self):
        inputs = zip(*raise_to_list(self.dataset.train_inputs))
        assert numpy.array_equal(inputs, [(1, 4), (2, 5)])
        targets = zip(*raise_to_list(self.dataset.train_targets))
        assert numpy.array_equal(targets, [(2,), (3,)])

    def testValid(self):
        inputs = zip(*raise_to_list(self.dataset.valid_inputs))
        assert numpy.array_equal(inputs, [([2, 3],), ([5, 6],), ([8, 9],)])
        targets = zip(*raise_to_list(self.dataset.valid_targets))
        assert numpy.array_equal(targets, [([3, 4],), ([6, 7],), ([9, 0],)])

    def testTest(self):
        inputs = zip(*raise_to_list(self.dataset.test_inputs))
        assert numpy.array_equal(inputs, [([3, 4],), ([6, 7],), ([1, 2],), ([9, 0],)])
        targets = zip(*raise_to_list(self.dataset.test_targets))
        assert numpy.array_equal(targets, [([4, 5],), ([7, 8],), ([2, 3],), ([0, 1],)])



    def tearDown(self):
        del self.dataset


if __name__ == '__main__':
    unittest.main()
