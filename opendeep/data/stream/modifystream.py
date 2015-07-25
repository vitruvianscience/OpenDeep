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
