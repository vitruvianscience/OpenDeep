"""
Generic structure for a dataset, and common sub-classes to deal with files/urls or arrays from numpy/scipy (in memory).

Attributes
----------
TRAIN : int
    The integer representing the training dataset subset.
VALID : int
    The integer representing the validation dataset subset.
TEST : int
    The integer representing the testing dataset subset.

.. todo:: Add large dataset support with database connections, numpy.memmap, h5py, pytables
    (and in the future grabbing from pipelines like spark)

.. todo:: Add methods for cleaning data, like normalizing to mean0 std1, or scaling to [min,max].
"""
# TODO: add large dataset support with database connections, numpy.memmap, h5py, pytables
# TODO: (and in the future grabbing from pipelines like spark)
# TODO: Add methods for cleaning data, like normalizing to mean0 std1, or scaling to [min,max].

__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
import os
import shutil
# third party libraries
import numpy
# internal imports
from opendeep import dataset_shared
from opendeep.utils.file_ops import mkdir_p, get_file_type, download_file
import opendeep.utils.file_ops as files

log = logging.getLogger(__name__)

# variables for each subset of the dataset
TRAIN = 0
VALID = 1
TEST  = 2

def get_subset_strings(subset):
    """
    Converts the subset integer to a string representation of TRAIN, VALID, or TEST.

    Parameters
    ----------
    subset : int
        The integer specifying the subset.

    Returns
    -------
    str
        The string representation of the subset.
    """
    if subset is TRAIN:
        return 'TRAIN'
    elif subset is VALID:
        return 'VALID'
    elif subset is TEST:
        return 'TEST'
    else:
        return str(subset)


class Dataset(object):
    '''
    Default interface for a dataset object. At minimum, a Dataset needs to implement getSubset() and getDataShape().
    getSubset() returns the (input, label) pair of shared variables holding the specific subset of this dataset.
    getDataShape() returns the list of (#examples, dimensionality) tuples for the given subset of this dataset, where
    each tuple in the list could represent a sequence. If there are no sequences or the dataset is padded, it would
    just return a single tuple.
    '''
    def getSubset(self, subset):
        """
        This method returns the single tuple of (input_data, labels) shared variables holding the given subset.

        Parameters
        ----------
        subset : int
            The subset indicator. Integer assigned by this module's attributes.

        Returns
        -------
        tuple
            (x, y) tuple of shared variables holding the dataset inputs and labels. If there aren't any labels (it is
            and unsupervised dataset), it should return (x, None).
            If the subset doesn't exist, it should return (None, None)
        """
        log.critical('No getSubset method implemented for %s!', str(type(self)))
        raise NotImplementedError()

    def getDataShape(self, subset):
        '''
        This method returns the shape (or list of shapes in the case of sequences) for the given subset of the data.

        Parameters
        ----------
        subset : int
            The subset indicator. Integer assigned by this module's attributes.

        Returns
        -------
        list(tuple) or tuple
            The list of (#examples, dimensionality) tuples for the given subset of this dataset, where
            each tuple in the list could represent a sequence. If there are no sequences or the dataset is padded,
            it would just return a single tuple representing the shape of the subset.
        '''
        log.critical('No getDataShape method implemented for %s!', str(type(self)))
        raise NotImplementedError()


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
    def __init__(self, filename=None, source=None, dataset_dir='../../datasets'):
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
        self.filename    = filename
        self.source      = source
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
            if self.source is not None:
                if file_type is None:
                    download_success = download_file(self.source, dataset_location)
                    file_type = get_file_type(dataset_location)
            else:
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


class MemoryDataset(Dataset):
    '''
    Dataset object wrapper for something given in memory (numpy matrix, theano matrix). You pass the array_like objects
    containing the subset inputs and labels, and the getSubset and getDataShape methods are automatically implemented.

    Attributes
    ----------
    train_X : shared variable
        The training input variables.
    train_Y : shared variable
        The training input labels.
    valid_X : shared variable
        The validation input variables.
    valid_Y : shared variable
        The validation input labels.
    test_X : shared variable
        The testing input variables.
    test_Y : shared variable
        The testing input labels.
    '''

    def __init__(self, train_X, train_Y=None, valid_X=None, valid_Y=None, test_X=None, test_Y=None):
        log.info('Wrapping matrix from memory')
        # make sure the inputs are arrays
        train_X = numpy.array(train_X)
        self._train_shape = train_X.shape
        self.train_X = dataset_shared(train_X, name='memory_train_x', borrow=True)
        if train_Y is not None:
            try:
                self.train_Y = dataset_shared(numpy.array(train_Y), name='memory_train_y', borrow=True)
            except Exception as e:
                log.exception("COULD NOT CONVERT train_Y TO NUMPY ARRAY. EXCEPTION: %s", str(e))

        if valid_X is not None:
            try:
                valid_X = numpy.array(valid_X)
                self._valid_shape = valid_X.shape
                self.valid_X = dataset_shared(valid_X, name='memory_valid_x', borrow=True)
            except Exception as e:
                log.exception("COULD NOT CONVERT valid_X TO NUMPY ARRAY. EXCEPTION: %s", str(e))
        if valid_Y is not None:
            try:
                self.valid_Y = dataset_shared(numpy.array(valid_Y), name='memory_valid_y', borrow=True)
            except Exception as e:
                log.exception("COULD NOT CONVERT valid_Y TO NUMPY ARRAY. EXCEPTION: %s", str(e))

        if test_X is not None:
            try:
                test_X = numpy.array(test_X)
                self._test_shape = test_X.shape
                self.test_X = dataset_shared(test_X, name='memory_test_x', borrow=True)
            except Exception as e:
                log.exception("COULD NOT CONVERT test_X TO NUMPY ARRAY. EXCEPTION: %s", str(e))
        if test_Y is not None:
            try:
                self.test_Y = dataset_shared(numpy.array(test_Y), name='memory_test_y', borrow=True)
            except Exception as e:
                log.exception("COULD NOT CONVERT test_Y TO NUMPY ARRAY. EXCEPTION: %s", str(e))

    def getSubset(self, subset):
        """
        Returns the (x, y) pair of shared variables for the given train, validation, or test subset.

        Parameters
        ----------
        subset : int
            The subset indicator. Integer assigned by global variables in opendeep.data.dataset.py

        Returns
        -------
        tuple
            (x, y) tuple of shared variables holding the dataset input and label, or None if the subset doesn't exist.
        """
        y = None
        if subset is TRAIN:
            if hasattr(self, 'train_Y'):
                y = self.train_Y
            return self.train_X, y
        elif subset is VALID and hasattr(self, 'valid_X') and self.valid_X:
            if hasattr(self, 'valid_Y'):
                y = self.valid_Y
            return self.valid_X, y
        elif subset is TEST and hasattr(self, 'test_X') and self.test_X:
            if hasattr(self, 'test_Y'):
                y = self.test_Y
            return self.test_X, y
        else:
            return None, None

    def getDataShape(self, subset):
        '''
        Returns the shape of the input data for the given subset

        Parameters
        ----------
        subset : int
            The subset indicator. Integer assigned by global variables in opendeep.data.dataset.py

        Returns
        -------
        tuple
            Return the shape of this dataset's subset in a (N, D) tuple where N=#examples and D=dimensionality
        '''
        if subset not in [TRAIN, VALID, TEST]:
            log.error('Subset %s not recognized!', get_subset_strings(subset))
            return (0, None)
        if subset is TRAIN:
            return self._train_shape
        elif subset is VALID and hasattr(self, '_valid_shape'):
            return self._valid_shape
        elif subset is TEST and hasattr(self, '_test_shape'):
            return self._test_shape
