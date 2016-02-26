"""
Generic structure for a dataset reading from a file or directory.
"""
# standard libraries
import logging
import os
# theano imports
from six import string_types
# internal imports
from opendeep.data.dataset import Dataset
from opendeep.data.stream.filestream import FileStream
from opendeep.utils.file_ops import install

_log = logging.getLogger(__name__)

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
    def __init__(self, path, source=None, train_filter=None, valid_filter=None, test_filter=None,
                 inputs_preprocess=None, targets_preprocess=None):
        """
        Creates a new :class:`FileDataset` from the path. It installs the file from the source
        if it isn't found in the path, and determines the filetype and full path location to the file.

        Parameters
        ----------
        path : str
            The name of the file or directory for the dataset.
        source : str, optional
            The URL path for downloading the dataset (if applicable).
        train_filter : regex string or compiled regex object or Tuple(regex string/compiled regex), optional
            The regular expression filter to match training file names against (if applicable). If given a tuple
            of two regular expressions, the first will be applied for inputs while the second will be applied
            for targets.
        valid_filter : regex string or compiled regex object or Tuple(regex string/compiled regex), optional
            The regular expression filter to match validation file names against (if applicable). If given a tuple
            of two regular expressions, the first will be applied for inputs while the second will be applied
            for targets.
        test_filter : regex string or compiled regex object or Tuple(regex string/compiled regex), optional
            The regular expression filter to match testing file names against (if applicable). If given a tuple
            of two regular expressions, the first will be applied for inputs while the second will be applied
            for targets.
        inputs_preprocess : function, optional
            A preprocessing function to apply to input data. This function will be applied to each line
            from the files in `path`, and if it creates a list of elements, each element will be yielded as the
            input data separately. For example, the function could be ``lambda line: (line.split(',')[0]).lower()``
            to grab a string before a comma on each line and lowercase it.
        targets_preprocess : function, optional
            A preprocessing function to apply to targets data. This function will be applied to each line from
            the files in `path`, and if it creates a list of elements, each element will be yielded as the target
            label data separately. For example, the function could be ``lambda line: (line.split(',')[1]).lower()``
            to grab a label after a comma on each line and lowercase it.
        """
        try:
            self.path = os.path.realpath(path)
        except Exception:
            _log.exception("Error creating os path for FileDataset from path %s" % self.path)
            raise

        self.source = source

        # install the dataset from source! (makes sure file is there and returns the type so you know how to read it)
        self.file_type = install(self.path, self.source)

        # preprocess functions
        self.inputs_preprocess = inputs_preprocess
        self.targets_preprocess = targets_preprocess

        # deal with tuples of filters
        train_inputs_filter, train_targets_filter = self._get_filters(train_filter)
        valid_inputs_filter, valid_targets_filter = self._get_filters(valid_filter)
        test_inputs_filter, test_targets_filter   = self._get_filters(test_filter)

        # inputs and targets
        train_inputs, train_targets = self._get_filestream(train_inputs_filter, train_targets_filter, is_train=True)
        valid_inputs, valid_targets = self._get_filestream(valid_inputs_filter, valid_targets_filter)
        test_inputs, test_targets   = self._get_filestream(test_inputs_filter, test_targets_filter)

        super(FileDataset, self).__init__(train_inputs=train_inputs, train_targets=train_targets,
                                          valid_inputs=valid_inputs, valid_targets=valid_targets,
                                          test_inputs=test_inputs, test_targets=test_targets)

    def _get_filters(self, filters):
        """
        Helper method to make sure the filter is a string or compiled regex object, or an iterable containing two
        strings or compiled regex objects.
        """
        filter1, filter2 = None, None
        # if it is a string, there is only 1 filter
        if isinstance(filters, string_types):
            filter1 = filters
        else:
            # try to unpack into two filters
            try:
                filter1, filter2 = filters
            # if ValueError (too many items to unpack), warn the exception
            except ValueError as e:
                _log.warn("ValueError: %s. Setting the filter to the input %s", e.message, str(filters))
                filter1 = filters
            # if TypeError (single non-iterable object), just set the first filter to the object
            except TypeError as e:
                filter1 = filters

        if filter2 is None and self.targets_preprocess is not None:
            filter2 = filter1

        return filter1, filter2

    def _get_filestream(self, inputs_filter, targets_filter, is_train=False):
        """
        Helper method to return the inputs and targets Filestream
        """
        inputs, targets = None, None

        # inputs filestream
        if inputs_filter is not None or is_train:
            inputs = FileStream(self.path, inputs_filter, self.inputs_preprocess)
        # targets filestream
        if targets_filter is not None or (is_train and self.targets_preprocess is not None):
            targets = FileStream(self.path, targets_filter, self.targets_preprocess)

        return inputs, targets
