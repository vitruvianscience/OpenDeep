from __future__ import division
import unittest
from opendeep.utils.batch import *
from opendeep.utils.misc import numpy_one_hot
import numpy

class TestBatch(unittest.TestCase):
    def setUp(self):
        # numpy array to test
        self.np = numpy.eye(10)
        # generator over word vectors to test
        words = "Hello\nWorld\nThis\nIs\nA\nTest!".split("\n")
        vocab = {char: n for char, n in zip(list(set(words)), range(len(set(words))))}
        words = [vocab[x] for x in words]
        self.words = numpy_one_hot(words, n_classes=len(vocab))

    def testNumpyBatchSize(self):
        for size in range(12):
            test = True
            try:
                batch = numpy_minibatch(self.np, batch_size=size)
                i = 0
                for x in batch:
                    assert x.shape[0] <= size
                    i += 1
            except AssertionError:
                assert size == 0
                test = False
            if test:
                iters = int(numpy.ceil(self.np.shape[0] / size))
                assert i == iters, "iterations was %d, expected %d for batch_size %d" % (i, iters, size)

    def testNumpyMinBatchSize(self):
        batch = numpy_minibatch(self.np, batch_size=3, min_batch_size=2)
        i = 0
        for x in batch:
            assert 2 <= x.shape[0] <= 3
            i += 1
        assert i == 3

        try:
            batch = numpy_minibatch(self.np, batch_size=2, min_batch_size=3)
            raise AssertionError("Was able to create batch with invalid sizes.")
        except Exception as e:
            assert isinstance(e, AssertionError)

    def testIterBatchSize(self):
        for size in range(12):
            gen = (row for row in self.words)
            test = True
            try:
                batch = iterable_minibatch(gen, batch_size=size)
                i = 0
                for x in batch:
                    assert x.shape[0] <= size
                    i += 1
            except AssertionError:
                assert size == 0
                test = False
            if test:
                iters = int(numpy.ceil(6. / size))
                assert i == iters, "iterations was %d, expected %d for batch_size %d" % (i, iters, size)

    def testIterMinBatchSize(self):
        gen = (row for row in self.words)
        batch = iterable_minibatch(gen, batch_size=4, min_batch_size=3)
        i = 0
        for x in batch:
            assert 3 <= x.shape[0] <= 4
            i += 1
        assert i == 1

        gen = (row for row in self.words)
        try:
            batch = iterable_minibatch(gen, batch_size=2, min_batch_size=3)
            raise AssertionError("Was able to create batch with invalid sizes.")
        except Exception as e:
            assert isinstance(e, AssertionError)

    def tearDown(self):
        del self.np, self.words


if __name__ == '__main__':
    unittest.main()
