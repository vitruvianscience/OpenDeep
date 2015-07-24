"""
A wrapper object for modifying iterable streams of data.
"""

class ModifyStream:
    """
    Creates an iterable stream modified from a source stream.

    Parameters
    ----------
    stream : iterable
        The input stream to modify.
    func : function
        The function to transform the input stream to an output stream (applied element-wise).
    """
    def __init__(self, stream, func):
        self.stream = stream
        self.func = func

    def __iter__(self):
        for elem in self.stream:
            yield self.func(elem)

class BufferStream:
    """
    Creates an iterable stream that returns a list of elements from the input stream with a buffer size.
    """
    def __init__(self, stream, buffer_size):
        self.stream = stream
        self.buffer_size = buffer_size

    def __iter__(self):
        pass