"""
This module provides an iterator that gives batches from a Numpy array.
"""
from __future__ import division

class NumpyBatches(object):
    def __init__(self, np_array, batch_size, min_batch_size):
        self.array = np_array
        assert min_batch_size <= batch_size, "batch_size (%d) has to be larger than min_batch_size (%d)!" % \
                                             (batch_size, min_batch_size)
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size

    def __iter__(self):
        for i in xrange(self.array.shape[0]//self.batch_size + 1):
            idx = i*self.batch_size
            data = self.array[idx:idx+self.batch_size]
            if data.shape[0] >= self.min_batch_size:
                yield data
