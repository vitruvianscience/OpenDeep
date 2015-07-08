# standard libraries
from __future__ import print_function
import unittest
import itertools
import numpy as np
# internal references
from opendeep.data.dataset_file import TextDataset
from opendeep.data.dataset import TRAIN, VALID, TEST
from opendeep.log.logger import config_root_logger


class TestCharsDataset(unittest.TestCase):

    def setUp(self):
        # configure the root logger
        config_root_logger()
        # get a logger for this session
        # create dataset
        shakespeare = '../../../datasets/shakespeare_input.txt'
        self.dataset = TextDataset(path=shakespeare,
                                   level="char",
                                   preprocess=lambda s: s.lower())

    def testExamples(self):
        print(self.dataset.vocab)
        n_examples = 100
        i = 0
        chars, labels = self.dataset.get_subset(TRAIN)
        for char, label in itertools.izip(chars, labels):
            if i > n_examples:
                break
            print(self.dataset.vocab_inverse[np.argmax(char)], self.dataset.vocab_inverse[np.argmax(label)])
            i += 1

    def tearDown(self):
        del self.dataset


if __name__ == '__main__':
    unittest.main()
