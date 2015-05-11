"""
These are basic utilities for working with files and filepaths.

Attributes
----------
DIRECTORY : int
    Directory marker.
ZIP : int
    Zip file marker.
GZ : int
    GZ file marker.
PKL : int
    Pickle file marker.
TAR : int
    Tarfile marker.
NPY : int
    Numpy save file marker.
UNKNOWN : int
    Unknown file type marker.
"""

__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

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
NPY       = 5
UNKNOWN   = 6


def get_filetype_string(filetype):
    """
    Given an integer depicting the filetype, return the string of the attribute.

    Parameters
    ----------
    filetype : int
        The filetype attribute from this class to get the string name.

    Returns
    -------
    str
        The string representation such as 'ZIP', 'PKL', 'UNKNOWN', etc.
    """
    if filetype is DIRECTORY:
        return 'DIRECTORY'
    elif filetype is ZIP:
        return 'ZIP'
    elif filetype is GZ:
        return 'GZ'
    elif filetype is PKL:
        return 'PKL'
    elif filetype is TAR:
        return 'TAR'
    elif filetype is NPY:
        return 'NPY'
    elif filetype is UNKNOWN:
        return 'UNKNOWN'
    else:
        return str(filetype)


def mkdir_p(path):
    """
    This function will create a filesystem path if it doesn't already exist (like mkdir in unix).

    Parameters
    ----------
    path : str
        The filesystem path to create.

    Raises
    ------
    OSError
        If there was an OS error making the directory.
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

    Parameters
    ----------
    filename : str
        The file to create (initialize as an empty file).
    """
    with open(filename, 'w') as f:
        f.write("")


def download_file(url, destination):
    """
    This will download whatever is on the internet at 'url' and save it to 'destination'.

    Parameters
    ----------
    url : str
        The URL to download from.
    destination : str
        The filesystem path (including file name) to download the file to.

    Returns
    -------
    bool
        Whether or not the operation was successful.
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
    Given a filename, try to determine the type of file from the extension into one of the categories defined as
    global variables above.
    Currently, can be .zip, .gz, .tar, .pkl, .p, or .pickle.

    Parameters
    ----------
    file_path : str
        The filesystem path to the file in question.

    Returns
    -------
    int
        The integer code to the file type defined in file_ops.py, or None if the file doesn't exist.
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
            elif extension == '.npy':
                return NPY
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

    Parameters
    ----------
    source_filename : str
        Filesystem path to the file to unzip.
    destination_dir : str
        Filesystem directory path for the file to unzip into.

    Returns
    -------
    bool
        Whether or not it was successful.
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

    Parameters
    ----------
    source_filename : str
        Filesystem path to the file to un-tar.
    destination_dir : str
        Filesystem path for the file to un-tar into.

    Returns
    -------
    bool
        Whether or not it was successful.
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