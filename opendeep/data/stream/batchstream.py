"""
A wrapper object for modifying iterable streams of data into batches/minibatches.
"""
# standard libraries
import itertools
# third party libraries
import numpy
# internal imports
from opendeep.utils.misc import raise_to_list

class BufferStream:
    """
    Creates an iterable stream that returns a list of elements from the input stream with a buffer size.
    """
    def __init__(self, stream, buffer_size):
        self.stream = stream
        self.buffer_size = buffer_size

    def __iter__(self):
        buffer = []
        for elem in self.stream:
            buffer.append(elem)
            if len(buffer) >= self.buffer_size:
                result = buffer
                buffer = []
                yield result

class MinibatchStream:
    """
    Creates a list of iterable streams of minibatches from an input list of streams.
    """
    def __init__(self, streams, batch_size, min_batch_size=1):
        self.streams = raise_to_list(streams)
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size

    def __iter__(self):
        iters = [iter(stream) for stream in self.streams]
        while True:
            chunks = [list(itertools.islice(it, self.batch_size)) for it in iters]
            # if there was nothing returned by any slice, return (assures stops at shortest stream)
            if any([len(chunk) == 0 for chunk in chunks]):
                return
            # otherwise if the chunk is above the acceptable minimum size, yield it as a numpy array!
            elif all(len(chunk) >= self.min_batch_size for chunk in chunks):
                # make sure they are all the same length
                min_len = min([len(chunk) for chunk in chunks])
                yield [numpy.asarray(chunk[:min_len]) for chunk in chunks]
