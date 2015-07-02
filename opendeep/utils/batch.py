"""
This module provides functions for working with batches.
"""
from __future__ import division, generators
# standard libraries
import logging
import itertools
# third party libraries
import numpy

log = logging.getLogger(__name__)

def minibatch(iterable, batch_size=1, min_batch_size=1):
    """
    This processes an iterable and yields batches of data of a given size (with a minimum size requirement).

    Parameters
    ----------
    iterable : iterator
        An iterable object (a numpy array is also iterable).
    batch_size : int, optional
        The number of examples to pull from the iterable as a batch. Default is 1.
    min_batch_size : int, optional
        The minimum number of examples to pull from the iterable. Default is 1.

    Yields
    ------
    numpy array
        A numpy array of the minibatch grabbed from the iterable.
    """
    assert 0 < min_batch_size <= batch_size, \
        "batch_size (%d) has to be larger than min_batch_size (%d) and they both have to be greater than zero!" % \
        (batch_size, min_batch_size)
    # if our input is just a numpy array, use the faster minibatching function
    if isinstance(iterable, numpy.ndarray):
        # would prefer to use 'yield from' but that syntax is python >= 3.3
        for chunk in numpy_minibatch(iterable, batch_size, min_batch_size):
            yield chunk
    # otherwise for general iterators, use the generic minibatching function.
    else:
        for chunk in iterable_minibatch(iterable, batch_size, min_batch_size):
            yield chunk

def iterable_minibatch(iterable, batch_size=1, min_batch_size=1):
    """
    This processes an iterable and yields batches of data of a given size (with a minimum size requirement).

    Parameters
    ----------
    iterable : iterator
        An iterable object.
    batch_size : int, optional
        The number of examples to pull from the iterable as a batch. Default is 1.
    min_batch_size : int, optional
        The minimum number of examples to pull from the iterable. Default is 1.

    Yields
    ------
    numpy array
        A numpy array of the minibatch grabbed from the iterable.
    """
    assert 0 < min_batch_size <= batch_size, \
        "batch_size (%d) has to be larger than min_batch_size (%d) and they both have to be greater than zero!" % \
        (batch_size, min_batch_size)

    # solution modified from http://stackoverflow.com/questions/8991506/iterate-an-iterator-by-chunks-of-n-in-python
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, batch_size))
        # if there was nothing returned by the slice, return
        if len(chunk) == 0:
            return
        # otherwise if the chunk is above the acceptable minimum size, yield it as a numpy array!
        elif len(chunk) >= min_batch_size:
            yield numpy.asarray(chunk)

def numpy_minibatch(numpy_array, batch_size=1, min_batch_size=1):
    """
    Creates a minibatch generator over a numpy array. :func:`minibatch` delegates to this generator
    when the input is a numpy.ndarray.

    Parameters
    ----------
    numpy_array : numpy.ndarray
        A numpy array.
    batch_size : int, optional
        The number of examples to pull from the array as a batch. Default is 1.
    min_batch_size : int, optional
        The minimum number of examples to pull from the iterable. Default is 1.

    Yields
    ------
    numpy array
        A numpy array of the minibatch. It will yield over the first dimension of the input.
    """
    numpy_array = numpy.asarray(numpy_array)
    assert 0 < min_batch_size <= batch_size, \
        "batch_size (%d) has to be larger than min_batch_size (%d) and they both have to be greater than zero!" % \
        (batch_size, min_batch_size)
    # go through the first dimension of the input array.
    for i in xrange((numpy_array.shape[0] // batch_size) + 1):
        idx = i * batch_size
        data = numpy_array[idx:(idx + batch_size)]
        if data.shape[0] >= min_batch_size:
            yield data
