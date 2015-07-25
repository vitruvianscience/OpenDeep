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
    log.debug('Making directory %s', path)
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

def install(path, source=None):
    """
    Method to both download and extract the dataset from the internet (if applicable) or verify that the file
    exists as ``path``.

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
    log.info('Installing dataset %s', str(path))
    is_dir = os.path.isdir(path)

    # construct the actual path to the dataset
    if is_dir:
        dataset_dir = os.path.splitext(path)[0]
    else:
        dataset_dir = os.path.split(path)[0]
    # make the directory or file directory
    try:
        mkdir_p(dataset_dir)
    except Exception as e:
        log.error("Couldn't make the dataset path with directory %s and path %s",
                  dataset_dir,
                  str(path))
        log.exception("%s", str(e.message))
        raise

    # check if the dataset is already in the source, otherwise download it.
    # first check if the base filename exists - without all the extensions.
    # then, add each extension on and keep checking until the upper level, when you download from http.
    if not is_dir:
        (dirs, fname) = os.path.split(path)
        split_fname = fname.split('.')
        accumulated_name = split_fname[0]
        ext_idx = 1
        while not found and ext_idx <= len(split_fname):
            if os.path.exists(os.path.join(dirs, accumulated_name)):
                found = True
                file_type = get_file_type(os.path.join(dirs, accumulated_name))
                path = os.path.join(dirs, accumulated_name)
                log.debug('Found file %s', path)
            elif ext_idx < len(split_fname):
                accumulated_name = '.'.join((accumulated_name, split_fname[ext_idx]))
            ext_idx += 1
    elif os.listdir(path):
        found = True
        file_type = get_file_type(path)

    # if the file wasn't found, download it if a source was provided. Otherwise, raise error.
    download_dest = None
    if not found:
        if source is not None and file_type is None:
            # make the destination for downloading the file from source be the same as self.path,
            # but make sure the source filename is preserved so we can deal with the appropriate
            # type (i.e. if it is a .tar.gz or a .zip, we have to process it first to match what
            # was expected with the real self.path filename)
            url_filename = source.split('/')[-1]
            download_dest = os.path.join(dataset_dir, url_filename)
            download_success = download_file(url=source, destination=download_dest)
            file_type = get_file_type(download_dest)
        elif source is None and file_type is None:
            log.error("Filename %s couldn't be found, and no URL source to download was provided.",
                      str(path))
            raise RuntimeError(
                "Filename %s couldn't be found, and no URL source to download was provided." %
                str(path))

    # if the file type is a zip and different than self.path, unzip it.
    unzip_success = True
    if file_type is not get_file_type(path) and download_dest is not None:
        unzip_success = False
        if file_type is ZIP:
            unzip_success = unzip(download_dest, dataset_dir)
        elif file_type is TARBALL or file_type is TAR:
            unzip_success = untar(download_dest, dataset_dir)
        elif file_type is GZ:
            unzip_success = gunzip(download_dest, dataset_dir)
        # if the unzip was successful
        if unzip_success:
            # remove the zipfile and update the dataset location and file type
            log.debug('Removing file %s', download_dest)
            os.remove(download_dest)
            file_type = get_file_type(path)

    if download_success and unzip_success:
        log.info('Installation complete. Yay!')
    else:
        log.warning('Something went wrong installing dataset. Boo :(')

    return file_type
