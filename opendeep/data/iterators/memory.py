"""
This module provides an iterator that gives batches from an array in memory.
"""
from __future__ import division
import numpy

class NumpyBatches(object):
    def __init__(self, arr, batch_size=1, min_batch_size=1):
        """
        :class:`opendeep.data.iterators.memory_batches.NumpyBatches` takes an array-like object and
        returns a batched iterator over it.

        Parameters
        ----------
        arr : array
            An array-like object (numpy array, etc.) to create the iterator over.
        batch_size : int, optional
            The size of batches to yield during iteration. This will always be over the first dimension in the array.
            Default is 1.
        min_batch_size : int, optional
            The minimum size of batches to yield during iteration. Default is 1.
        """
        self.array = numpy.asarray(arr)
        assert min_batch_size <= batch_size, "batch_size (%d) has to be larger than min_batch_size (%d)!" % \
                                             (batch_size, min_batch_size)
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size

    def __iter__(self):
        """
        Fulfills the iterator requirement to call .next().

        Yields
        ------
        numpy array
            A batch of data from the initialization array. Batches are done over the first dimension of the array.
        """
        for i in xrange((self.array.shape[0] // self.batch_size) + 1):
            idx = i*self.batch_size
            data = self.array[idx:idx+self.batch_size]
            if data.shape[0] >= self.min_batch_size:
                yield data
