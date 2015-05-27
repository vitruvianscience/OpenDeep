# standard libraries
from __future__ import print_function
import unittest
import logging
import numpy as np
# internal references
from opendeep.data.standard_datasets.text.characters import CharsLM
import opendeep.data.dataset as dataset
import opendeep.log.logger as logger


class TestCharsDataset(unittest.TestCase):

    def setUp(self):
        # configure the root logger
        logger.config_root_logger()
        # get a logger for this session
        self.log = logging.getLogger(__name__)
        # create dataset
        self.dataset = CharsLM(filename='shakespeare_input.txt', seq_length=500, train_split=0.9, valid_split=0.05)

    def testExamples(self):
        train = self.dataset.getSubset(dataset.TRAIN)[0]
        valid = self.dataset.getSubset(dataset.VALID)[0]
        test = self.dataset.getSubset(dataset.TEST)[0]

        print("Total # sequences:", self.dataset.length)
        print("Train sequences:", self.dataset._train_len)
        print("Valid sequences:", self.dataset._valid_len)
        print("Test sequences:", self.dataset._test_len)

        train = np.where(train.get_value()[:1])[2]
        if valid:
            valid = np.where(valid.get_value()[:1])[2]
        else:
            valid = []
        if test:
            test = np.where(test.get_value()[:1])[2]
        else:
            test = []

        inverse = {v: k for k, v in self.dataset.vocab.items()}
        train = [inverse[i] for i in train]
        valid = [inverse[i] for i in valid]
        test  = [inverse[i] for i in test]
        print("-----------------------train-----------------------------")
        print(''.join(train))
        if len(valid) > 0:
            print("-----------------------valid-----------------------------")
            print(''.join(valid))
        if len(test) > 0:
            print("-----------------------test-----------------------------")
            print(''.join(test))

    def tearDown(self):
        del self.dataset


if __name__ == '__main__':
    unittest.main()