"""
.. module:: dataset

Generic structure for a dataset, and common sub-classes to deal with files/urls or arrays from numpy/scipy (in memory).
"""
# TODO: add large dataset support with database connections, numpy.memmap, h5py, pytables (and in the future grabbing from pipelines like spark)

__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import logging
import os
import shutil
# third party libraries
import numpy
# internal imports
from opendeep import sharedX
from opendeep.utils.file_ops import mkdir_p, get_file_type, download_file
import opendeep.utils.file_ops as files

log = logging.getLogger(__name__)

# variables for each subset of the dataset
TRAIN = 0
VALID = 1
TEST  = 2

def get_subset_strings(subset):
    if subset is TRAIN:
        return 'TRAIN'
    elif subset is VALID:
        return 'VALID'
    elif subset is TEST:
        return 'TEST'
    else:
        return str(subset)

# TODO: I don't think this is very efficient implementation, especially with the iterators. However, it is flexible. Need to look into it further to optimize.
class Dataset(object):
    '''
    Default interface for a dataset object - a bunch of sources for an iterator to grab data from
    '''
    def __init__(self):
        pass

    def getDataByIndices(self, indices, subset):
        '''
        This method is used by an iterator to return data values at given indices.
        :param indices: either integer or list of integers
        The index (or indices) of values to return
        :param subset: integer
        The integer representing the subset of the data to consider dataset.(TRAIN, VALID, or TEST)
        :return: array
        The dataset values at the index (indices)
        '''
        log.critical('No getDataByIndices method implemented for %s!', str(type(self)))
        raise NotImplementedError()


    def getLabelsByIndices(self, indices, subset):
        '''
        This method is used by an iterator to return data label values at given indices.
        :param indices: either integer or list of integers
        The index (or indices) of values to return
        :param subset: integer
        The integer representing the subset of the data to consider dataset.(TRAIN, VALID, or TEST)
        :return: array
        The dataset labels at the index (indices)
        '''
        log.critical('No getLabelsByIndices method implemented for %s!', str(type(self)))
        raise NotImplementedError()


    def hasSubset(self, subset):
        '''
        :param subset: integer
        The integer representing the subset of the data to consider dataset.(TRAIN, VALID, or TEST)
        :return: boolean
        Whether or not this dataset has the given subset split
        '''
        if subset not in [TRAIN, VALID, TEST]:
            log.error('Subset %s not recognized!', get_subset_strings(subset))
        if subset is TRAIN:
            return True
        elif subset is VALID and hasattr(self, '_valid_shape'):
            return True
        elif subset is TEST and hasattr(self, '_test_shape'):
            return True
        else:
            return False


    def getDataShape(self, subset):
        '''
        :return: tuple
        Return the shape of this dataset's subset in a NxD tuple where N=#examples and D=dimensionality
        '''
        if subset not in [TRAIN, VALID, TEST]:
            log.error('Subset %s not recognized!', get_subset_strings(subset))
        if subset is TRAIN and hasattr(self, '_train_shape'):
            return self._train_shape
        elif subset is VALID and hasattr(self, '_valid_shape'):
            return self._valid_shape
        elif subset is TEST and hasattr(self, '_test_shape'):
            return self._test_shape
        else:
            log.critical('No getDataShape method implemented for %s for subset %s!', str(type(self)), get_subset_strings(subset))
            raise NotImplementedError()


    def get_example_shape(self):
        return self.getDataShape(TRAIN)[1]


    def scaleMeanZeroVarianceOne(self):
        '''
        Scale the dataset input vectors X to have a mean of 0 and variance of 1
        :return: boolean
        Whether successful
        '''
        log.critical('No scaleMeanZeroVarianceOne method implemented for %s', str(type(self)))
        raise NotImplementedError()


    def scaleMinMax(self, min, max):
        '''
        Scale the dataset input vectors X to have a minimum and maximum
        :param min: integer
        Minimum value in input vector X
        :param max: integer
        Maximum value in input vector X
        :return: boolean
        Whether successful
        '''
        log.critical('No scaleMinMax method implemented for %s', str(type(self)))
        raise NotImplementedError()


class FileDataset(Dataset):
    '''
    Default interface for a file-based dataset object - a bunch of sources for an iterator to grab data from
    '''
    def __init__(self, filename=None, source=None, dataset_dir='../../datasets'):
        super(FileDataset, self).__init__()

        self.filename    = filename
        self.source      = source
        self.dataset_dir = dataset_dir

        # install the dataset from source! (makes sure the file is there and returns the type so you know how to read it)
        self.dataset_location, self.file_type = self.install()

    # helper methods for installing dataset files
    def install(self):
        '''
        Method to both download and extract the dataset from the internet (if there) or verify connection settings
        '''
        file_type=None
        if self.filename is not None and self.source is not None:
            log.info('Installing dataset %s', str(self.source))
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

            # if the file wasn't found, download it.
            download_success = True
            if file_type is None:
                download_success = download_file(self.source, dataset_location)
                file_type = get_file_type(dataset_location)

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
        Method to delete dataset files from disk (if in file form)
        '''
        # TODO: Check if this shutil.rmtree is unsafe...
        log.info('Uninstalling (removing) dataset %s...', self.dataset_location)
        if self.dataset_location is not None and os.path.exists(self.dataset_location):
            # If we are trying to remove something not from the dataset directory, give a warning
            if not self.dataset_location.startswith(self.dataset_dir):
                log.critical("ATTEMPTING TO REMOVE A FILE NOT FROM THE DATASET DIRECTORY. LOCATION IS %s AND THE DATASET DIRECTORY IS %s",
                             self.dataset_location,
                             self.dataset_dir)
            shutil.rmtree(self.dataset_location)
        else:
            log.debug('dataset_location was not valid. It was %s', str(self.dataset_location))
        log.info('Uninstallation (removal) successful!')


class MemoryDataset(Dataset):
    '''
    Dataset object wrapper for something given in memory (numpy matrix, theano matrix)
    '''

    def __init__(self, train_X, train_Y=None, valid_X=None, valid_Y=None, test_X=None, test_Y=None):
        log.info('Wrapping matrix from memory')
        super(self.__class__, self).__init__()

        # make sure the inputs are arrays
        train_X = numpy.array(train_X)
        self._train_shape = train_X.shape
        self.train_X = sharedX(train_X)
        if train_Y:
            self.train_Y = sharedX(numpy.array(train_Y))

        if valid_X:
            valid_X = numpy.array(valid_X)
            self._valid_shape = valid_X.shape
            self.valid_X = sharedX(valid_X)
        if valid_Y:
            self.valid_Y = sharedX(numpy.array(valid_Y))

        if test_X:
            test_X = numpy.array(test_X)
            self._test_shape = test_X.shape
            self.test_X = sharedX(test_X)
        if test_Y:
            self.test_Y = sharedX(numpy.array(test_Y))

    def getDataByIndices(self, indices, subset):
        '''
        This method is used by an iterator to return data values at given indices.
        :param indices: either integer or list of integers
        The index (or indices) of values to return
        :param subset: integer
        The integer representing the subset of the data to consider dataset.(TRAIN, VALID, or TEST)
        :return: array
        The dataset values at the index (indices)
        '''
        if subset is TRAIN:
            return self.train_X.get_value(borrow=True)[indices]
        elif subset is VALID and hasattr(self, 'valid_X') and self.valid_X:
            return self.valid_X.get_value(borrow=True)[indices]
        elif subset is TEST and hasattr(self, 'test_X') and self.test_X:
            return self.test_X.get_value(borrow=True)[indices]
        else:
            return None

    def getLabelsByIndices(self, indices, subset):
        '''
        This method is used by an iterator to return data label values at given indices.
        :param indices: either integer or list of integers
        The index (or indices) of values to return
        :param subset: integer
        The integer representing the subset of the data to consider dataset.(TRAIN, VALID, or TEST)
        :return: array
        The dataset labels at the index (indices)
        '''
        if subset is TRAIN and hasattr(self, 'train_Y') and self.train_Y:
            return self.train_Y.get_value(borrow=True)[indices]
        elif subset is VALID and hasattr(self, 'valid_Y') and self.valid_Y:
            return self.valid_Y.get_value(borrow=True)[indices]
        elif subset is TEST and hasattr(self, 'test_Y') and self.test_Y:
            return self.test_Y.get_value(borrow=True)[indices]
        else:
            return None