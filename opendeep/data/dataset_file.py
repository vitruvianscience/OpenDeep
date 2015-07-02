"""
Generic structure for a dataset reading from files.
"""

# standard libraries
import logging
import os
import shutil
# internal imports
from opendeep.data.dataset import Dataset
from opendeep.utils.file_ops import mkdir_p, get_file_type, download_file
import opendeep.utils.file_ops as files

log = logging.getLogger(__name__)

class FileDataset(Dataset):
    '''
    Default interface for a file-based dataset object. Files should either exist in the dataset_dir or have
    a downloadable source. Subclasses should implement the specific methods for extracting data from their
    respective files.

    Attributes
    ----------
    filename : str
        The name of the file for the dataset.
    source : str
        The URL path for downloading the dataset (if applicable).
    dataset_dir : str
        The base path to the folder for containing the dataset filename.
    dataset_location : str
        The full path to the dataset on disk.
    file_type : int
        The integer representing the type of file for this dataset. The file_type integer is assigned by the
        opendeep.utils.file_ops module.
    '''
    def __init__(self, filenames=None, sources=None, dataset_dir='../../datasets'):
        """
        Creates a new FileDataset from the filename, source, and dataset_dir. It installs the file from the source
        if it isn't found in the dataset_dir, and determines the filetype and full path location to the file.

        Parameters
        ----------
        filename : str
            The name of the file for the dataset.
        source : str
            The URL path for downloading the dataset (if applicable).
        dataset_dir : str
            The base path to the folder for containing the dataset filename.
        """
        self.filenames    = filenames
        self.sources      = sources
        self.dataset_dir = dataset_dir

        # install the dataset from source! (makes sure file is there and returns the type so you know how to read it)
        self.dataset_location, self.file_type = self.install()

    def install(self):
        '''
        Method to both download and extract the dataset from the internet (if applicable) or verify that the file
        exists in the dataset_dir.

        Returns
        -------
        str
            The absolute path to the dataset location on disk.
        int
            The integer representing the file type for the dataset, as defined in the opendeep.utils.file_ops module.
        '''
        file_type = None
        if self.filename is not None:
            log.info('Installing dataset %s', str(self.filename))
            # construct the actual path to the dataset
            prevdir = os.getcwd()
            os.chdir(os.path.split(os.path.realpath(__file__))[0])
            dataset_dir = os.path.realpath(self.dataset_dir)
            try:
                mkdir_p(dataset_dir)
                dataset_location = os.path.join(dataset_dir, self.filename)
            except Exception as e:
                log.error("Couldn't make the dataset path with directory %s and filename %s",
                          dataset_dir,
                          str(self.filename))
                log.exception("%s", str(e))
                dataset_location = None
            finally:
                os.chdir(prevdir)

            # check if the dataset is already in the source, otherwise download it.
            # first check if the base filename exists - without all the extensions.
            # then, add each extension on and keep checking until the upper level, when you download from http.
            if dataset_location is not None:
                (dirs, fname) = os.path.split(dataset_location)
                split_fname = fname.split('.')
                accumulated_name = split_fname[0]
                found = False
                # first check if the filename was a directory (like for the midi datasets)
                if os.path.exists(os.path.join(dirs, accumulated_name)):
                    found = True
                    file_type = get_file_type(os.path.join(dirs, accumulated_name))
                    dataset_location = os.path.join(dirs, accumulated_name)
                    log.debug('Found file %s', dataset_location)
                # now go through the file extensions starting with the lowest level and check if the file exists
                if not found and len(split_fname) > 1:
                    for chunk in split_fname[1:]:
                        accumulated_name = '.'.join((accumulated_name, chunk))
                        file_type = get_file_type(os.path.join(dirs, accumulated_name))
                        if file_type is not None:
                            dataset_location = os.path.join(dirs, accumulated_name)
                            log.debug('Found file %s', dataset_location)
                            break

            # if the file wasn't found, download it if a source was provided. Otherwise, raise error.
            download_success = True
            if self.source is not None and file_type is None:
                download_success = download_file(self.source, dataset_location)
                file_type = get_file_type(dataset_location)
            elif self.source is None and file_type is None:
                log.error("Filename %s couldn't be found, and no URL source to download was provided.",
                          str(self.filename))
                raise RuntimeError("Filename %s couldn't be found, and no URL source to download was provided." %
                                   str(self.filename))

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
                    file_type = get_file_type(dataset_location)
            if download_success and unzip_success:
                log.info('Installation complete. Yay!')
            else:
                log.warning('Something went wrong installing dataset. Boo :(')

            return dataset_location, file_type

    def uninstall(self):
        '''
        Method to delete dataset files from its dataset_location on disk.

        .. warning::
            This method currently uses the shutil.rmtree method, which may be unsafe. A better bet would be to delete
            the dataset yourself from disk.
        '''
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
