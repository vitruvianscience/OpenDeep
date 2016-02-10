"""
A wrapper object for generators of data from files.
"""
import logging
try:
    from PIL import Image
    has_pil = True
except ImportError:
    has_pil = False
import numpy
from opendeep.utils.file_ops import find_files
from opendeep.utils.misc import raise_to_list

_log = logging.getLogger(__name__)

class FileStream:
    """
    Creates an iterable stream of data from a filepath of text-based files.

    Parameters
    ----------
    path : str or iterable(str)
        The filesystem path to stream, or an iterable of filenames.
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
        for fname in find_files(self.path, self.filter):
            try:
                with open(fname, 'r') as f:
                    for line in f:
                        if self.preprocess is not None and callable(self.preprocess):
                            line = self.preprocess(line)
                        line = raise_to_list(line)
                        for token in line:
                            if idx >= self.n_future:
                                yield token
                            else:
                                idx += 1
            except Exception as err:
                _log.exception(err.__str__())

class FilepathStream:
    """
    Creates an iterable stream from filepath names in a path. This is just for the file names themselves, not the
    contents of the files. If you want the contents, use the :class:`FileStream` object.

    Parameters
    ----------
    path : str or iterable(str)
        The filesystem path to stream, or an iterable of filenames.
    filter : str or compiled regex, optional
        The regex filter to apply to the `path` when finding files.
    preprocess : function, optional
        A function to apply to the names of files found in the `path`. If a list is returned from
        the preprocess function, each element will be yielded separately during iteration.
    """
    def __init__(self, path, filter=None, preprocess=None):
        self.path = path
        self.filter = filter
        self.preprocess = preprocess

    def __iter__(self):
        for fname in find_files(self.path, self.filter):
            if self.preprocess is not None and callable(self.preprocess):
                fname = self.preprocess(fname)
            fnames = raise_to_list(fname)
            for name in fnames:
                yield name

class ImageStream:
    """
    Creates an iterable stream of data from a filepath of image files.

    Parameters
    ----------
    path : str or iterable(str)
        The filesystem path to stream, or an iterable of filenames.
    filter : str or compiled regex, optional
        The regex filter to apply to the `path` when finding files.
    preprocess : function, optional
        A function to apply to the image returned from files found in the `path`. If a list is returned from
        the preprocess function, each element will be yielded separately during iteration.
    """
    def __init__(self, path, filter=None, preprocess=None):
        if not has_pil:
            raise NotImplementedError("You need the PIL (pillow) Python package to use ImageStream.")

        self.path = path
        self.filter = filter
        self.preprocess = preprocess

    def __iter__(self):
        for fname in find_files(self.path, self.filter):
            try:
                with Image.open(fname) as im:
                    data = numpy.array(im)
                    if self.preprocess is not None and callable(self.preprocess):
                        data = self.preprocess(data)
                    data = raise_to_list(data)
                    for d in data:
                        yield d
            except Exception as err:
                _log.exception(err.__str__())
