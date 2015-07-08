"""
Generic structure for a dataset reading from a file or directory.
"""
# standard libraries
import logging
import os
import shutil
import time
import warnings
# third party
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
# internal imports
from opendeep.data.dataset import Dataset, TRAIN, VALID, TEST
import opendeep.utils.file_ops as files
from opendeep.utils.decorators import inherit_docs
from opendeep.utils.misc import numpy_one_hot, make_time_units_string

log = logging.getLogger(__name__)

def _yield_readline(path, filter):
    """
    Finds the files in the path (with the regex filter applied) and yields their list of lines
    from readlines()

    Parameters
    ----------
    path : str
        System path to find the files
    filter : str or compiled regex
        Regex filter to apply to filepaths

    Yields
    ------
    Str
        Each line from each file found in the path
    """
    for fname in files.find_files(path, filter):
        with open(fname, 'rb') as f:
            for line in f:
                yield line


@inherit_docs
class FileDataset(Dataset):
    """
    Default interface for a file-based dataset object. Files should either exist in the ``path`` or have
    a downloadable source. Subclasses should implement the specific methods for extracting data from their
    respective files.

    Attributes
    ----------
    path : str
        The full location to the dataset file or directory on disk.
    source : str
        The URL path for downloading the dataset (if applicable).
    file_type : int
        The integer representing the type of file for this dataset. The file_type integer is assigned by the
        :mod:`opendeep.utils.file_ops` module.
    """
    def __init__(self, path, source=None, train_filter=None, valid_filter=None, test_filter=None):
        """
        Creates a new FileDataset from the path. It installs the file from the source
        if it isn't found in the path, and determines the filetype and full path location to the file.

        Parameters
        ----------
        path : str
            The name of the file or directory for the dataset.
        source : str, optional
            The URL path for downloading the dataset (if applicable).
        train_filter : regex string or compiled regex object, optional
            The regular expression filter to match training file names against (if applicable).
        valid_filter : regex string or compiled regex object, optional
            The regular expression filter to match validation file names against (if applicable).
        test_filter : regex string or compiled regex object, optional
            The regular expression filter to match testing file names against (if applicable).
        """
        try:
            self.path = os.path.realpath(path)
        except Exception:
            log.exception("Error creating os path for FileDataset from path %s" % self.path)
            raise

        self.source = source
        self.train_filter = train_filter
        self.valid_filter = valid_filter
        self.test_filter = test_filter

        # install the dataset from source! (makes sure file is there and returns the type so you know how to read it)
        self.file_type = self.install()

    def install(self):
        """
        Method to both download and extract the dataset from the internet (if applicable) or verify that the file
        exists as ``self.path``.

        Returns
        -------
        str
            The absolute path to the dataset location on disk.
        int
            The integer representing the file type for the dataset,
            as defined in the :mod:`opendeep.utils.file_ops` module.
        """
        file_type = None
        download_success = True
        found = False
        log.info('Installing dataset %s', str(self.path))
        is_dir = os.path.isdir(self.path)

        # construct the actual path to the dataset
        if is_dir:
            dataset_dir = os.path.splitext(self.path)[0]
        else:
            dataset_dir = os.path.split(self.path)[0]
        # make the directory or file directory
        try:
            files.mkdir_p(dataset_dir)
        except Exception as e:
            log.error("Couldn't make the dataset path with directory %s and path %s",
                      dataset_dir,
                      str(self.path))
            log.exception("%s", str(e.message))
            raise

        # check if the dataset is already in the source, otherwise download it.
        # first check if the base filename exists - without all the extensions.
        # then, add each extension on and keep checking until the upper level, when you download from http.
        if not is_dir:
            (dirs, fname) = os.path.split(self.path)
            split_fname = fname.split('.')
            accumulated_name = split_fname[0]
            ext_idx = 1
            while not found and ext_idx <= len(split_fname):
                if os.path.exists(os.path.join(dirs, accumulated_name)):
                    found = True
                    file_type = files.get_file_type(os.path.join(dirs, accumulated_name))
                    self.path = os.path.join(dirs, accumulated_name)
                    log.debug('Found file %s', self.path)
                elif ext_idx < len(split_fname):
                    accumulated_name = '.'.join((accumulated_name, split_fname[ext_idx]))
                ext_idx += 1
        elif os.listdir(self.path):
            found = True
            file_type = files.get_file_type(self.path)

        # if the file wasn't found, download it if a source was provided. Otherwise, raise error.
        download_dest = None
        if not found:
            if self.source is not None and file_type is None:
                # make the destination for downloading the file from source be the same as self.path,
                # but make sure the source filename is preserved so we can deal with the appropriate
                # type (i.e. if it is a .tar.gz or a .zip, we have to process it first to match what
                # was expected with the real self.path filename)
                url_filename = self.source.split('/')[-1]
                download_dest = os.path.join(dataset_dir, url_filename)
                download_success = files.download_file(url=self.source, destination=download_dest)
                file_type = files.get_file_type(download_dest)
            elif self.source is None and file_type is None:
                log.error("Filename %s couldn't be found, and no URL source to download was provided.",
                          str(self.path))
                raise RuntimeError(
                    "Filename %s couldn't be found, and no URL source to download was provided." %
                    str(self.path))

        # if the file type is a zip and different than self.path, unzip it.
        unzip_success = True
        if file_type is not files.get_file_type(self.path) and download_dest is not None:
            unzip_success = False
            if file_type is files.ZIP:
                unzip_success = files.unzip(download_dest, dataset_dir)
            elif file_type is files.TARBALL or file_type is files.TAR:
                unzip_success = files.untar(download_dest, dataset_dir)
            elif file_type is files.GZ:
                unzip_success = files.gunzip(download_dest, dataset_dir)
            # if the unzip was successful
            if unzip_success:
                # remove the zipfile and update the dataset location and file type
                log.debug('Removing file %s', download_dest)
                os.remove(download_dest)
                file_type = files.get_file_type(self.path)

        if download_success and unzip_success:
            log.info('Installation complete. Yay!')
        else:
            log.warning('Something went wrong installing dataset. Boo :(')

        return file_type

    def uninstall(self):
        """
        Method to delete dataset files from its path on disk.

        .. warning::
            This method currently uses the shutil.rmtree method, which may be unsafe. A better bet would be to delete
            the dataset yourself from disk.
        """
        # TODO: Check if this shutil.rmtree is unsafe...
        log.info('Uninstalling (removing) dataset %s', self.path)
        if self.path is not None and os.path.exists(self.path):
            # If we are trying to remove something not from the dataset directory, give a warning
            if not self.path.startswith(self.dataset_dir):
                log.critical("ATTEMPTING TO REMOVE A FILE NOT FROM THE DATASET DIRECTORY. "
                             "LOCATION IS %s AND THE DATASET DIRECTORY IS %s",
                             self.path,
                             self.dataset_dir)
            shutil.rmtree(self.path)
        else:
            log.debug('path was not valid. It was %s', str(self.path))
        log.info('Uninstallation (removal) successful!')

    def get_subset(self, subset):
        # return chained iterators over the files for the subset given the filters
        if subset is TRAIN:
            return _yield_readline(self.path, self.train_filter), None
        elif subset is VALID and self.valid_filter is not None:
            return _yield_readline(self.path, self.valid_filter), None
        elif subset is TEST and self.test_filter is not None:
            return _yield_readline(self.path, self.test_filter), None

@inherit_docs
class TextDataset(FileDataset):
    """
    This gives a file-based dataset for working with text (either characters or words).
    It will construct a vocabulary dictionary with each token.
    """
    def __init__(self, path, source=None, train_filter=None, valid_filter=None, test_filter=None,
                 vocab=None, unk_token="<UNK>", level="char", preprocess=None):
        """
        Initialize a text-based dataset.

        Parameters
        ----------
        path : str
            The name of the file or directory for the dataset.
        source : str, optional
            The URL path for downloading the dataset (if applicable).
        train_filter : regex string or compiled regex object, optional
            The regular expression filter to match training file names against (if applicable).
        valid_filter : regex string or compiled regex object, optional
            The regular expression filter to match validation file names against (if applicable).
        test_filter : regex string or compiled regex object, optional
            The regular expression filter to match testing file names against (if applicable).
        vocab : dict, optional
            A starting dictionary to use when converting tokens to numbers.
        unk_token : str, optional
            The representation for an unknown token to use in the vocab dictionary.
        level : str, optional
            Either ``char``, ``word``, or ``line``, saying how to process the text.
            For ``char``, data will be character-level.
            For ``word``, data will be split by whitespace.
            For ``line``, data will be split by newline.
        preprocess : function, optional
            A function to apply to data directly from path (i.e. lambda s: s.lower() for lowercase)
        """
        super(TextDataset, self).__init__(path=path, source=source,
                                          train_filter=train_filter, valid_filter=valid_filter, test_filter=test_filter)
        level = level.lower()
        levels = ["char", "word", "line"]
        assert level in levels, "level parameter needs to be one of %s, found %s!" % (str(levels), level)
        if level == "word":
            if not NLTK_AVAILABLE:
                warnings.warn("NLTK isn't installed - going to split strings by whitespace. Highly recommended "
                              "that you install nltk for better word tokenization.")
        self.unk_token = unk_token
        self.level = level
        self.preprocess = preprocess

        # Create our vocab dictionary if it doesn't exist!
        self.vocab = vocab or self.compile_vocab()
        self.vocab_inverse = {v:k for k,v in self.vocab.items()}

    def tokens(self, filter=None):
        """
        Returns all tokens in the path (with regex filter applied) given the level and preprocessing.

        Parameters
        ----------
        filter : regex string or compiled regex object, optional
            The regular expression filter to match file names against (if applicable).
        """
        for fname in files.find_files(self.path, filter):
            with open(fname, 'r') as f:
                for line in f:
                    if self.preprocess is not None:
                        line = self.preprocess(line)
                    if self.level == "char":
                        tokens = line
                    elif self.level == "word":
                        if NLTK_AVAILABLE:
                            tokens = nltk.tokenize.word_tokenize(line)
                        else:
                            tokens = line.split()
                    elif self.level == "line":
                        tokens = [line]
                    else:
                        tokens = []
                    for token in tokens:
                        yield token

    def compile_vocab(self):
        """
        Creates a dictionary mapping tokens (words or characters) to integers given the level and preprocessing.
        """
        log.debug("Creating vocabulary...")
        t = time.time()
        vocab = {self.unk_token: 0}
        i = 1
        for token in self.tokens():
            if token not in vocab:
                vocab[token] = i
                i += 1
        log.debug("Vocab took %s to create." % make_time_units_string(time.time() - t))
        return vocab

    def vectorize_tokens(self, filter=None):
        """
        Generates the one-hot vectorized tokens from the files.

        Parameters
        ----------
        filter : regex string or compiled regex object, optional
            The regular expression filter to match file names against (if applicable).
        """
        for token in self.tokens(filter):
            rep = self.vocab.get(token, self.vocab[self.unk_token])
            yield numpy_one_hot([rep], n_classes=len(self.vocab))[0]

    def get_subset(self, subset):
        if subset is TRAIN:
            # labels is the same generator as data, just advanced forward by one.
            labels_gen = self.vectorize_tokens(self.train_filter)
            # advance forward
            next(labels_gen)
            return self.vectorize_tokens(self.train_filter), labels_gen
        elif subset is VALID and self.valid_filter is not None:
            # labels is the same generator as data, just advanced forward by one.
            labels_gen = self.vectorize_tokens(self.valid_filter)
            # advance forward
            next(labels_gen)
            return self.vectorize_tokens(self.valid_filter), labels_gen
        elif subset is TEST and self.test_filter is not None:
            # labels is the same generator as data, just advanced forward by one.
            labels_gen = self.vectorize_tokens(self.test_filter)
            # advance forward
            next(labels_gen)
            return self.vectorize_tokens(self.test_filter), labels_gen
        else:
            return None, None
