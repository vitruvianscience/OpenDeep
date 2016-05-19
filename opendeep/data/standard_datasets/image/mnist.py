"""
Provides the MNIST handwritten digit dataset.

See: http://yann.lecun.com/exdb/mnist/
"""
# standard libraries
import logging
import gzip
# third party libraries
from numpy import concatenate, reshape
# internal imports
from opendeep.data.dataset_file import FileDataset
from opendeep.utils import file_ops
from opendeep.utils.misc import numpy_one_hot, binarize

try:
    import cPickle as pickle
except ImportError:
    import pickle

log = logging.getLogger(__name__)

# python 2 vs. python 3 mnist source
import sys
if sys.version_info > (3, 0):
    mnist_source = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist_py3k.pkl.gz'
    mnist_path = 'datasets/mnist_py3k.pkl.gz'
else:
    mnist_source = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    mnist_path = 'datasets/mnist.pkl.gz'

class MNIST(FileDataset):
    """
    Object for the MNIST handwritten digit dataset. Pickled file provided by Montreal's LISA lab into
    train, valid, and test sets. http://www.iro.umontreal.ca/~lisa/deep/data/mnist/
    """
    def __init__(self, flatten=False, binary_cutoff=False, one_hot=False, concat_train_valid=False,
                 path=mnist_path, source=mnist_source):
        """
        Parameters
        ----------
        flatten : bool, optional
            Flag to flatten the 2D images into 1D vectors.
        binary_cutoff : float, optional
            If you want to binarize the input images, what threshold value to use.
        one_hot : bool, optional
            Flag to convert the labels to one-hot encoding rather than their normal integers.
        concat_train_valid : bool, optional
            Flag to concatenate the training and validation datasets together. This would be the original split.
        path : str, optional
            The `path` parameter to a ``FileDataset``.
        source : str, optional
            The `source` parameter to a ``FileDataset``.
        """
        # instantiate the Dataset class to install the dataset from the url
        log.info("Loading MNIST with binary={!s} and one_hot={!s}".format(str(binary_cutoff), str(one_hot)))

        super(MNIST, self).__init__(path=path, source=source)

        # self.path now contains the os path to the dataset file
        # self.file_type tells how to load the dataset
        # load the dataset into memory
        if self.file_type is file_ops.GZ:
            (self.train_inputs, self.train_targets), \
            (self.valid_inputs, self.valid_targets), \
            (self.test_inputs, self.test_targets) = pickle.load(
                gzip.open(self.path, 'rb')
            )
        else:
            (self.train_inputs, self.train_targets), \
            (self.valid_inputs, self.valid_targets), \
            (self.test_inputs, self.test_targets) = pickle.load(
                open(self.path, 'r')
            )

        if concat_train_valid:
            log.debug("Concatenating train and valid sets together...")
            self.train_inputs = concatenate((self.train_inputs, self.valid_inputs))
            self.train_targets = concatenate((self.train_targets, self.valid_targets))

        # make optional binary
        if binary_cutoff:
            log.debug("Making MNIST input values binary with cutoff {!s}".format(str(binary_cutoff)))
            self.train_inputs = binarize(self.train_inputs, binary_cutoff)
            self.valid_inputs = binarize(self.valid_inputs, binary_cutoff)
            self.test_inputs  = binarize(self.test_inputs, binary_cutoff)

        # make optional one-hot labels
        if one_hot:
            self.train_targets = numpy_one_hot(self.train_targets, n_classes=10)
            self.valid_targets = numpy_one_hot(self.valid_targets, n_classes=10)
            self.test_targets  = numpy_one_hot(self.test_targets, n_classes=10)

        # This data source comes pre-flattened. If not flatten, then expand to (1, 28, 28)
        if not flatten:
            self.train_inputs = reshape(self.train_inputs, (self.train_inputs.shape[0], 1, 28, 28))
            self.valid_inputs = reshape(self.valid_inputs, (self.valid_inputs.shape[0], 1, 28, 28))
            self.test_inputs  = reshape(self.test_inputs, (self.test_inputs.shape[0], 1, 28, 28))

        log.debug("MNIST train shape: {!s}, {!s}".format(self.train_inputs.shape, self.train_targets.shape))
        log.debug("MNIST valid shape: {!s}, {!s}".format(self.valid_inputs.shape, self.valid_targets.shape))
        log.debug("MNIST test shape: {!s}, {!s}".format(self.test_inputs.shape, self.test_targets.shape))
