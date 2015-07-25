"""
A wrapper object for generators of data from files.
"""
import opendeep.utils.file_ops as files
from opendeep.utils.misc import raise_to_list

class FileStream:
    """
    Creates an iterable stream of tokens from a filepath.

    Parameters
    ----------
    path : str
        The filesystem path to stream.
    filter : str or compiled regex, optional
        The regex filter to apply to the `path` when finding files.
    preprocess : function, optional
        A function to apply to each line returned from files found in the `path`. If a list is returned from
        the preprocess function, each element will be yielded separately during iteration.
    n_future : int, optional
        The number of tokens to start in the future (from the beginning of the first file). This is used often
        when creating language models and you want the targets stream to start 1 or more tokens in the future
        compared to the inputs stream.
    """
    def __init__(self, path, filter=None, preprocess=None, n_future=None):
        self.path = path
        self.filter = filter
        self.preprocess = preprocess
        self.n_future = n_future or 0

    def __iter__(self):
        idx = 0
        for fname in files.find_files(self.path, self.filter):
            with open(fname, 'r') as f:
                for line in f:
                    if self.preprocess is not None:
                        line = self.preprocess(line)
                    line = raise_to_list(line)
                    for token in line:
                        if idx >= self.n_future:
                            yield token
                        else:
                            idx += 1
