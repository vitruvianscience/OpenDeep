"""
A wrapper object for modifying iterable streams of data into batches/minibatches.
"""
# standard libraries
import itertools
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
        pass