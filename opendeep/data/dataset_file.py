"""
Generic structure for a dataset reading from a file or directory.
"""
# standard libraries
import logging
import os
import shutil
# internal imports
from opendeep.data.dataset import Dataset
import opendeep.utils.file_ops as files
from opendeep.utils.decorators import inherit_docs

log = logging.getLogger(__name__)

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
        is_dir = os.path.isdir(self.dir)

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

            # if the file wasn't found, download it if a source was provided. Otherwise, raise error.
            if not found:
                if self.source is not None and file_type is None:
                    # make the destination for downloading the file from source be the same as self.path,
                    # but make sure the source filename is preserved so we can deal with the appropriate
                    # type (i.e. if it is a .tar.gz or a .zip, we have to process it first to match what
                    # was expected with the real self.path filename)
                    url_filename = self.source.split('/')[-1]
                    dest = os.path.join(dirs, url_filename)
                    download_success = files.download_file(url=self.source, destination=dest)
                    file_type = files.get_file_type(dest)
                elif self.source is None and file_type is None:
                    log.error("Filename %s couldn't be found, and no URL source to download was provided.",
                              str(self.path))
                    raise RuntimeError("Filename %s couldn't be found, and no URL source to download was provided." %
                                       str(self.path))
        # if it is a directory, check if it is empty
        else:


        # if the file type is a zip, unzip it.
        unzip_success = True
        if file_type is files.ZIP:
            (dirs, fname) = os.path.split(dataset_location)
            post_unzip = os.path.join(dirs, '.'.join(fname.split('.')[0:-1]))
            unzip_success = files.unzip(dataset_location, post_unzip)
            # if the unzip was successful
            if unzip_success:
                # remove the zipfile and update the dataset location and file type
                log.debug('Removing file %s', dataset_location)
                os.remove(dataset_location)
                dataset_location = post_unzip
                file_type = files.get_file_type(dataset_location)
        if download_success and unzip_success:
            log.info('Installation complete. Yay!')
        else:
            log.warning('Something went wrong installing dataset. Boo :(')

        return file_type

    def uninstall(self):
        """
        Method to delete dataset files from its dataset_location on disk.

        .. warning::
            This method currently uses the shutil.rmtree method, which may be unsafe. A better bet would be to delete
            the dataset yourself from disk.
        """
        # TODO: Check if this shutil.rmtree is unsafe...
        log.info('Uninstalling (removing) dataset %s', self.dataset_location)
        if self.dataset_location is not None and os.path.exists(self.dataset_location):
            # If we are trying to remove something not from the dataset directory, give a warning
            if not self.dataset_location.startswith(self.dataset_dir):
                log.critical("ATTEMPTING TO REMOVE A FILE NOT FROM THE DATASET DIRECTORY. "
                             "LOCATION IS %s AND THE DATASET DIRECTORY IS %s",
                             self.dataset_location,
                             self.dataset_dir)
            shutil.rmtree(self.dataset_location)
        else:
            log.debug('dataset_location was not valid. It was %s', str(self.dataset_location))
        log.info('Uninstallation (removal) successful!')

    def get_subset(self, subset):
        raise NotImplementedError()
