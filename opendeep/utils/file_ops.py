"""
.. module:: file_ops

These are basic utilities for working with files and filepaths.
"""

__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard imports
import os
import errno
import urllib
import zipfile
import tarfile
import logging

log = logging.getLogger(__name__)

# variables for file format types
DIRECTORY = 0
ZIP       = 1
GZ        = 2
PKL       = 3
TAR       = 4
UNKNOWN   = 5


def mkdir_p(path):
    """
    This function will create a filesystem path if it doesn't already exist (mkdir in unix)

    :param path: the filesystem path to create
    :type path: String

    :raises OSError
    """
    path = os.path.realpath(path)
    log.debug('Attempting to make directory %s', path)
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            log.debug('Directory %s already exists!', path)
            pass
        else:
            log.exception('Error making directory %s', path)
            raise


def init_empty_file(filename):
    """
    This function will create an empty file (containing an empty string) with the given filename. This is similar to
    the 'touch' command in unix.

    :param filename: the file to create
    :type filename: String
    """
    with open(filename, 'w') as f:
        f.write("")


def download_file(url, destination):
    """
    This will download whatever is on the internet at 'url' and save it to 'destination'.

    :param url: the URL to download from
    :type url: String

    :param destination: the filesystem path (including file name) to download the file to
    :type destination: String

    :return: whether or not the operation was successful
    :rtype: Boolean
    """
    destination = os.path.realpath(destination)
    log.debug('Downloading data from %s to %s', url, destination)
    try:
        page = urllib.urlopen(url)
        if page.getcode() is not 200:
            log.warning('Tried to download data from %s and got http response code %s', url, str(page.getcode()))
            return False
        urllib.urlretrieve(url, destination)
        return True
    except:
        log.exception('Error downloading data from %s to %s', url, destination)
        return False


def get_file_type(file_path):
    """
    Given a filename, try to determine the type of file from the extension into one of the categories defined as global variables above.
    Currently, can be .zip, .gz, .tar, .pkl, .p, or .pickle.

    :param file_path: the filesystem path to the file in question
    :type file_path: String

    :return: the integer code to the file type defined in file_ops.py, or None if the file doesn't exist.
    :rtype: Integer or None
    """
    file_path = os.path.realpath(file_path)
    if os.path.exists(file_path):
        # if it is a directory
        if os.path.isdir(file_path):
            return DIRECTORY
        # otherwise if it is a file
        elif os.path.isfile(file_path):
            _, extension = os.path.splitext(file_path)
            extension = extension.lower()
            if extension == '.zip':
                return ZIP
            elif extension == '.gz':
                return GZ
            elif extension == '.tar':
                return TAR
            elif extension == '.pkl' or extension == '.p' or extension =='.pickle':
                return PKL
            else:
                log.warning('Didn\'t recognize file extension %s for file %s', extension, file_path)
                return UNKNOWN
        else:
            log.warning('File %s isn\'t a file or directory, but it exists... WHAT ARE YOU?!?', file_path)
            return UNKNOWN
    else:
        log.debug('File %s doesn\'t exist!', file_path)
        return None


def unzip(source_filename, destination_dir='.'):
    """
    This will unzip a source file (.zip) to a destination directory.

    :param source_filename: Filesystem path to the file to unzip
    :type source_filename: String

    :param destination_dir: Filesystem directory path for the file to unzip into
    :type destination_dir: String

    :return: Whether or not it was successful
    :rtype: Boolean
    """
    source_filename = os.path.realpath(source_filename)
    destination_dir = os.path.realpath(destination_dir)
    log.debug('Unzipping data from %s to %s', source_filename, destination_dir)
    try:
        with zipfile.ZipFile(source_filename) as zf:
            zf.extractall(destination_dir)
            return True
    except:
        log.exception('Error unzipping data from %s to %s', source_filename, destination_dir)
        return False


def untar(source_filename, destination_dir='.'):
    """
    This will unzip a tarball (.tar.gz) to a destination directory.

    :param source_filename: Filesystem path to the file to un-tar
    :type source_filename: String

    :param destination_dir: Filesystem path for the file to un-tar into
    :type destination_dir: String

    :return: Whether or not it was successful
    :rtype: Boolean
    """
    source_filename = os.path.realpath(source_filename)
    destination_dir = os.path.realpath(destination_dir)
    log.debug('Unzipping tarball data from %s to %s', source_filename, destination_dir)
    try:
        with tarfile.open(source_filename) as tar:
            tar.extractall(destination_dir)
            return True
    except:
        log.exception('Error unzipping tarball data from %s to %s', source_filename, destination_dir)
        return False