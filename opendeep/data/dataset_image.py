"""
Generic structure for a dataset reading from a file or directory.
"""
# standard libraries
import logging
import os
# internal imports
from opendeep.data.dataset import Dataset
from opendeep.data.stream.filestream import ImageStream
import opendeep.utils.file_ops as files

log = logging.getLogger(__name__)

class ImageDataset(Dataset):
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
            log.exception("Error creating os path for ImageDataset from path %s" % self.path)
            raise

        self.source = source

        # install the dataset from source! (makes sure file is there and returns the type so you know how to read it)
        self.file_type = files.install(self.path, self.source)

        train_inputs, train_targets = None, None
        valid_inputs, valid_targets = None, None
        test_inputs, test_targets   = None, None

        train_inputs = ImageStream(self.path, train_filter, inputs_preprocess)
        if targets_preprocess is not None:
            train_targets = ImageStream(self.path, train_filter, targets_preprocess)

        if valid_filter is not None:
            valid_inputs = ImageStream(self.path, valid_filter, inputs_preprocess)
            if targets_preprocess is not None:
                valid_targets = ImageStream(self.path, valid_filter, targets_preprocess)

        if test_filter is not None:
            test_inputs = ImageStream(self.path, test_filter, inputs_preprocess)
            if targets_preprocess is not None:
                test_targets = ImageStream(self.path, test_filter, targets_preprocess)

        super(ImageDataset, self).__init__(train_inputs=train_inputs, train_targets=train_targets,
                                           valid_inputs=valid_inputs, valid_targets=valid_targets,
                                           test_inputs=test_inputs, test_targets=test_targets)
