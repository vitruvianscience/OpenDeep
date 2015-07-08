"""
These are basic utilities for working with files and filenames.

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
TXT : int
    Text file marker.
UNKNOWN : int
    Unknown file type marker.
"""
# standard imports
import os
import errno
import urllib
import zipfile
import tarfile
import logging
import re
import gzip
# third party
from theano.compat.six import string_types

log = logging.getLogger(__name__)

# variables for file format types
DIRECTORY = 0
ZIP       = 1
TAR       = 2
GZ        = 3
TARBALL   = 4
PKL       = 5
NPY       = 6
TXT       = 7
UNKNOWN   = 8
_types = {
    DIRECTORY: "DIRECTORY",
    ZIP: "ZIP",
    GZ: "GZ",
    PKL: "PKL",
    TAR: "TAR",
    TARBALL: "TARBALL",
    NPY: "NPY",
    TXT: "TXT",
    UNKNOWN: "UNKNOWN"
}

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
    return _types.get(filetype, str(filetype))

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

def find_files(path, path_filter=None):
    """
    Recursively walks directories in ``path`` (if it is a directory) to find the files that have names
    matching ``path_filter``.

    Parameters
    ----------
    path : str
        The path to the directory to walk or file to find.
    path_filter : regular expression string or compiled regular expression object
        The regular expression to match against file path names.
    """
    if path_filter is not None:
        if isinstance(path_filter, string_types):
            reg = re.compile(path_filter)
        else:
            reg = path_filter
    else:
        reg = None

    path = os.path.realpath(path)
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for basename in files:
                filepath = os.path.join(root, basename)
                try:
                    if reg is None or reg.match(filepath) is not None:
                        yield filepath
                except TypeError as te:
                    log.exception("TypeError exception when finding files. %s" % str(te.message))
                    raise
    elif os.path.isfile(path):
        try:
            if reg is None or reg.match(path) is not None:
                yield path
        except TypeError as te:
            log.exception("TypeError exception when finding files. %s" % str(te.message))
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
    Currently, can be .zip, .gz, .tar, .tar.gz, .pkl, .p, .pickle, or .txt.

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
            fname, extension = os.path.splitext(file_path)
            extension = extension.lower()
            if extension == '.zip':
                return ZIP
            elif extension == '.gz':
                # check if tarball
                _, ext2 = os.path.splitext(fname)
                if ext2:
                    ext2 = ext2.lower()
                    if ext2 == '.tar':
                        return TARBALL
                # otherwise just gz
                return GZ
            elif extension == '.tar':
                return TAR
            elif extension == '.pkl' or extension == '.p' or extension =='.pickle':
                return PKL
            elif extension == '.npy':
                return NPY
            elif extension == '.txt':
                return TXT
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

def gunzip(source_filename, destination_dir='.'):
    """
    This will unzip a .gz to a destination directory.

    Parameters
    ----------
    source_filename : str
        Filesystem path to the file to gunzip.
    destination_dir : str
        Filesystem path for the file to gunzip into.

    Returns
    -------
    bool
        Whether or not it was successful.
    """
    source_filename = os.path.realpath(source_filename)
    destination_dir = os.path.realpath(destination_dir)
    log.debug('Unzipping gz data from %s to %s', source_filename, destination_dir)
    fpath, _ = os.path.splitext(source_filename)
    _, fname = os.path.split(fpath)
    dest_file = os.path.join(destination_dir, fname)
    try:
        with gzip.open(source_filename, 'rb') as gz:
            data = gz.read()
            with open(dest_file, 'wb') as f:
                f.write(data)
            return True
    except:
        log.exception('Error unzipping gz data from %s to %s', source_filename, destination_dir)
        return False