# standard libraries
from __future__ import print_function
import unittest
import os
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import numpy as np
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
# internal references
from opendeep.data import TextDataset

class TestCharsDataset(unittest.TestCase):

    def setUp(self):
        # create dataset
        self.shakespeare = 'shakespeare_input.txt'
        self.n_chars = 20
        with open(self.shakespeare, 'w') as f:
            f.write("This is a test\n\nCitizen: wherefore art thou\nI don't know!\n\nAren't we going to just"
                    "\nkill ourselves anyway? That is how these things normally work.\n\nOh, I know. Tis the "
                    "way of life, really. Is this a long\n\n\nenough test? Why did I just format that oddly?\n"
                    "Who knows, it is the inspiration from writing tests!")
        with open(self.shakespeare, 'r') as f:
            self.first_n_chars = list(f.read(self.n_chars))

    def testLanguageModel(self):
        for n_future in [0, 1, 10, 13]:
            dataset = TextDataset(path=self.shakespeare,
                                  level="char",
                                  target_n_future=n_future)
            i = 0
            chars, labels = dataset.train_inputs, dataset.train_targets
            for char, label in zip(chars, labels):
                char = dataset.vocab_inverse[np.argmax(char, 0)]
                label = dataset.label_vocab_inverse[np.argmax(label, 0)]
                if i >= self.n_chars - n_future:
                    break
                assert char == self.first_n_chars[i], \
                    "Expected %s at index %d, found %s" % (self.first_n_chars[i], i, char)
                assert label == self.first_n_chars[i+n_future], \
                    "Expected label %s at index %d, found %s" % (self.first_n_chars[i+n_future], i, label)
                i += 1
            del dataset

        for len in [2, 5]:
            dataset = TextDataset(path=self.shakespeare,
                                  level="char",
                                  target_n_future=1,
                                  sequence_length=len)

            chars, labels = dataset.train_inputs, dataset.train_targets
            for i, (char_seq, label_seq) in enumerate(zip(chars, labels)):
                char_s = [dataset.vocab_inverse[np.argmax(char, 0)] for char in char_seq]
                label_s = [dataset.label_vocab_inverse[np.argmax(label, 0)] for label in label_seq]

            del dataset

    def testLevels(self):
        # char
        dataset = TextDataset(path=self.shakespeare,
                              level="char")
        for i, char in enumerate(dataset.train_inputs):
            if i >= self.n_chars:
                break
            char = dataset.vocab_inverse[np.argmax(char)]
            assert char == self.first_n_chars[i], \
                "Expected %s at index %d, found %s" % (self.first_n_chars[i], i, char)

        # word
        dataset = TextDataset(path=self.shakespeare,
                              level="word")
        for i, word, in enumerate(dataset.train_inputs):
            if i >= self.n_chars:
                break
            print(dataset.vocab_inverse[np.argmax(word)])

        # line
        with open(self.shakespeare, 'r') as f:
            lines = f.readlines()[:self.n_chars]
        print(lines)
        dataset = TextDataset(path=self.shakespeare,
                              level="line")
        for i, line in enumerate(dataset.train_inputs):
            if i>= self.n_chars:
                break
            print([dataset.vocab_inverse[np.argmax(line)]])

    def tearDown(self):
        os.remove(self.shakespeare)


if __name__ == '__main__':
    unittest.main()
