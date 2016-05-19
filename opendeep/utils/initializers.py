"""
Provides various methods for initializing shared variables.

Based on code from Li Yao (University of Montreal)
https://github.com/yaoli/GSN

And theano_alexnet (https://github.com/uoguelph-mlrg/theano_alexnet)
"""
# standard libraries
import logging
from functools import partial
from types import FunctionType
from collections import Iterable
# third party libraries
import numpy
from theano import config
from theano.tensor import TensorVariable
from six import string_types
# internal imports
from opendeep.utils.constructors import as_floatX, sharedX

log = logging.getLogger(__name__)

numpy.random.RandomState(23455)
# set a fixed number initializing RandomSate for 2 purpose:
#  1. repeatable experiments; 2. for multiple-GPU, the same initial weights

number_types = (int, long, float, numpy.number)


class UniformIntervalFunc():
    """
    A collection of static functions for computing an interval to use for :class:`Uniform` initialization.
    Each function must be a result of an input shape tuple.
    """
    @staticmethod
    def default(shape):
        """
        Default interval calculation in a few codebases
        shape[0] = rows

        Parameters
        ----------
        shape : tuple
            Input shape to make weights matrix.

        Returns
        -------
        float
            1 / sqrt(shape[0])
        """
        return 1 / numpy.sqrt(shape[0])

    @staticmethod
    def glorot(shape):
        """
        Most common initialization scheme in literature.

        shape[0] = rows
        shape[1] = cols
        numpy.prod(shape[2:]) = receptive field from Glorot et al.

        Parameters
        ----------
        shape : tuple
            Input shape to make weights matrix.

        Returns
        -------
        float
            sqrt(6. / ((shape[0] + shape[1]) * prod(shape[2:])))
        """
        return numpy.sqrt(6. / ((shape[0] + shape[1]) * numpy.prod(shape[2:])))

    @staticmethod
    def sigmoid(shape):
        """
        Use this only when the activation function is sigmoid

        shape[0] = rows
        shape[1] = cols
        numpy.prod(shape[2:]) = receptive field from Glorot et al.

        Parameters
        ----------
        shape : tuple
            Input shape to make weights matrix.

        Returns
        -------
        float
            4 * sqrt(6. / ((shape[0] + shape[1]) * prod(shape[2:])))
        """
        return 4 * numpy.sqrt(6. / ((shape[0] + shape[1]) * numpy.prod(shape[2:])))


class Initializer(object):
    """
    The :class:`Initializer` is basically a partially applied function. It implements a __call__ method which
    allows you to call the instantiated class to construct a shared variable with a given shape under the
    Initializer's specific function. For example, this could be a uniform random number generator, where the
    initialization parameters could be high and low range, and the __call__ method returns a shared variable
    given the uniform random sample function.
    """
    def __call__(self, shape, name=None):
        """
        Parameters
        ----------
        shape : int or tuple
            The shape to use for the construction of the shared variable given the initializer function.
        name : str, optional
            The Initializer must have a name for the shared variable construction. This could be W for weights or
            b for biases.

        Returns
        -------
        Shared Variable
            The shared variable with name: `name` containing the initial values with shape `shape`.
        """
        raise NotImplementedError("Initializer {!s} not implemented!".format(self.__class__.__name__))


class Uniform(Initializer):
    """
    The :class:`Uniform` is an :class:`Initializer` that will take a uniform random sample
    over an interval given the shape provided when the class is called.
    """
    def __init__(self, interval, gain=1., rng=numpy.random):
        """
        Parameters
        ----------
        interval : str, float, tuple(float), or function(shape)
            The interval to use for the minimum and maximum numbers when drawing from uniform distribution.
            If it is a string, look up the appropriate function to generate interval in :class:`UniformIntervalFunc`.
            If it is a float (or number type), use [-abs(interval), abs(interval)]
            If it is a tuple (or Iterable), use [numpy.min(interval), numpy.max(interval)]
            If it is a function, use the float or tuple methods given the return value of the function evaluated on
            `shape` with the __call__ method is invoked for this class.
        gain : float, optional
            A multiplicative factor to affect the whole weights matrix.
        rng : random, optional
            A given random number generator to use (must have a .uniform function).
        """
        self.gain = gain
        self.rng = rng
        if not hasattr(self.rng, "uniform"):
            msg = "`rng` input to Uniform does not have `uniform` attribute."
            log.error(msg)
            raise AttributeError(msg)

        if isinstance(interval, string_types):
            if hasattr(UniformIntervalFunc, interval):
                self.interval_func = getattr(UniformIntervalFunc, interval)
            else:
                import inspect
                msg = "Interval with name {!s} can't be found in the UniformIntervalFunc class. " \
                      "Please try one of: {!s}".format(
                    interval,
                    [name for name, obj in inspect.getmembers(UniformIntervalFunc)
                    if not name.startswith("__")]
                )
                log.error(msg)
                raise AttributeError(msg)
        elif isinstance(interval, number_types):
            self._parse_single_number(interval)
        elif isinstance(interval, Iterable):
            self._parse_tuple(interval)
        elif isinstance(interval, FunctionType):
            self.interval_func = interval
        else:
            msg = "Bad argument `interval` to Uniform. Found {!s} of type {!s}".format(str(interval), type(interval))
            log.error(msg)
            raise AttributeError(msg)

    def _parse_single_number(self, interval):
        interval = abs(interval)
        self.min = -1*interval
        self.max = interval

    def _parse_tuple(self, interval):
        interval = list(interval)
        for element in interval:
            if not isinstance(element, number_types):
                msg = "Bad tuple element in Uniform. Found type {!s}. Full list: {!s}".format(type(element), interval)
                log.error(msg)
                raise AttributeError(msg)
        if len(interval) == 1:
            self._parse_single_number(interval[0])
        elif len(interval) > 1:
            self.min = numpy.min(interval)
            self.max = numpy.max(interval)
        else:
            msg = "`interval` tuple length not >=1. Found {!s}".format(len(interval))
            log.error(msg)
            raise AttributeError(msg)

    def __call__(self, shape, name=None):
        """
        Create the shared variable with given shape from the uniform distribution with interval described in __init__.

        Parameters
        ----------
        shape : tuple
            A tuple giving the shape information for this variable.
        name : str, optional
            The name to give the shared variable.

        Returns
        -------
        shared variable
            The shared variable with given shape and name drawn from a uniform distribution.
        """
        # if the min and max are determined by a function of shape
        if hasattr(self, "interval_func"):
            try:
                interval = self.interval_func(shape)
                if isinstance(interval, number_types):
                    self._parse_single_number(interval)
                elif isinstance(interval, Iterable):
                    self._parse_tuple(interval)
            except Exception as err:
                msg = "Expected interval function to output a number or Iterable of numbers, found {!s}".format(type(interval))
                log.error(msg)
                raise AttributeError(msg)

        # build the uniform weights tensor
        log.debug("Creating variable {!s} with shape {!s} from Uniform interval [{!s}, {!s}]".format(
            name, shape, self.min, self.max
        ))
        val = as_floatX(self.rng.uniform(low=self.min, high=self.max, size=shape))
        # check if a theano rng was used
        if isinstance(val, TensorVariable):
            val = val.eval()
        # multiply by gain factor
        if self.gain != 1.:
            log.debug("Multiplying {!s} by {!s}".format(name, self.gain))
        val = val * self.gain
        # make it into a shared variable
        return sharedX(value=val, name=name)


class Gaussian(Initializer):
    """
    The :class:`Gaussian` an :class:`Initializer` that will take a gaussian (normal) random sample with provided
    mean and standard deviation when the class is called.
    """
    def __init__(self, mean=0, std=0.05, gain=1., rng=numpy.random):
        """
        Parameters
        ----------
        mean : float
            Mean to use in Gaussian distribution.
        std : float
            Standard deviation to use in Gaussian distribution
        gain : float
            A multiplicative factor to affect the whole matrix.
        rng : random
            A given random number generator to use with .normal method.
        """
        self.mean = mean
        self.std = std
        self.gain = gain
        self.rng = rng
        if not hasattr(self.rng, "normal"):
            msg = "`rng` input to Gaussian does not have `normal` attribute."
            log.error(msg)
            raise AttributeError(msg)

    def __call__(self, shape, name=None):
        """
        Create the shared variable with given shape from the Gaussian distribution described in __init__.

        Parameters
        ----------
        shape : tuple
            A tuple giving the shape information for this variable.
        name : str, optional
            The name to give the shared variable.

        Returns
        -------
        shared variable
            The shared variable with given shape and name drawn from a Gaussian (normal) distribution.
        """
        log.debug("Creating variable {!s} with shape {!s} from Gaussian mean={!s}, std={!s}".format(
            name, shape, self.mean, self.std
        ))
        if self.std != 0:
            if isinstance(self.rng, type(numpy.random)):
                val = numpy.asarray(self.rng.normal(loc=self.mean, scale=self.std, size=shape), dtype=config.floatX)
            else:
                val = numpy.asarray(self.rng.normal(avg=self.mean, std=self.std, size=shape).eval(), dtype=config.floatX)
        else:
            val = as_floatX(self.mean * numpy.ones(shape, dtype=config.floatX))

        # check if a theano rng was used
        if isinstance(val, TensorVariable):
            val = val.eval()
        # multiply by gain factor
        if self.gain != 1.:
            log.debug("Multiplying {!s} by {!s}".format(name, self.gain))
        val = val * self.gain
        # make it into a shared variable
        return sharedX(value=val, name=name)


class Identity(Initializer):
    """
    The :class:`Identity` an :class:`Initializer` that will return a matrix as close to the identity as possible.
    If a non-square shape, it will make a matrix of the form (I 0)

    Identity matrix for weights is useful for RNNs with ReLU! http://arxiv.org/abs/1504.00941
    """
    def __init__(self, add_noise=None, gain=1.):
        """
        Parameters
        ----------
        add_noise : functools.partial, optional
            A partially applied noise function (just missing the `input` parameter) to add noise to the identity
            initialization. Noise functions can be found in opendeep.utils.noise.
        gain : float, optional
            A multiplicative factor to affect the whole weights matrix.
        """
        self.gain = gain
        self.add_noise = add_noise

    def __call__(self, shape, name=None):
        """
        Create the shared variable with given shape as an Identity matrix.

        Parameters
        ----------
        shape : tuple
            A tuple giving the shape information for this variable.
        name : str, optional
            The name to give the shared variable.

        Returns
        -------
        shared variable
            The shared variable with given shape and name as an Identity matrix.
        """
        log.debug("Creating variable {!s} with shape {!s} as Identity".format(name, shape))
        weights = numpy.eye(N=shape[0], M=int(numpy.prod(shape[1:])), k=0, dtype=config.floatX)

        if self.add_noise:
            if isinstance(self.add_noise, partial):
                weights = self.add_noise(input=weights)
            else:
                log.error("Add noise to identity weights was not a functools.partial object. Ignoring...")
        # multiply by gain factor
        if self.gain != 1.:
            log.debug("Multiplying {!s} by {!s}".format(name, self.gain))
        val = weights * self.gain
        return sharedX(value=val, name=name)


class Orthogonal(Initializer):
    """
    The :class:`Orthogonal` an :class:`Initializer` that will return orthonormal random values to
    initialize a weight matrix (using SVD).

    Some discussion here:
    http://www.reddit.com/r/MachineLearning/comments/2qsje7/how_do_you_initialize_your_neural_network_weights/

    From Lasagne:
    For n-dimensional shapes where n > 2, the n-1 trailing axes are flattened.
    For convolutional layers, this corresponds to the fan-in, so this makes the initialization
    usable for both dense and convolutional layers.
    """
    def __init__(self, gain=1., rng=numpy.random):
        """
        Parameters
        ----------
        gain : float
            A multiplicative factor to affect the whole weights matrix.
        rng : random
            A given random number generator to use with .normal method.
        """
        self.gain = gain
        self.rng = rng
        if not hasattr(self.rng, "normal"):
            msg = "`rng` input to Orthogonal does not have `normal` attribute."
            log.error(msg)
            raise AttributeError(msg)

    def __call__(self, shape, name=None):
        """
        Parameters
        ----------
        shape : tuple
            Tuple giving the shape information for the weight matrix.
        name : str
            Name to give the shared variable.

        Returns
        -------
        shared variable
            The shared variable orthogonal matrix with given shape.
        """
        log.debug("Creating Orthogonal matrix weights {!s} with shape {!s}".format(name, shape))
        if len(shape) == 1:
            shape = (shape[0], shape[0])
        else:
            # flatten shapes bigger than 2
            # From Lasagne: For n-dimensional shapes where n > 2, the n-1 trailing axes are flattened.
            # For convolutional layers, this corresponds to the fan-in, so this makes the initialization
            # usable for both dense and convolutional layers.
            shape = (shape[0], numpy.prod(shape[1:]))

        # Sample from the standard normal distribution
        if isinstance(self.rng, type(numpy.random)):
            a = numpy.asarray(self.rng.normal(loc=0., scale=1., size=shape), dtype=config.floatX)
        else:
            a = numpy.asarray(self.rng.normal(avg=0., std=1., size=shape).eval(), dtype=config.floatX)

        u, _, _ = numpy.linalg.svd(a, full_matrices=False)

        # multiply by gain factor
        if self.gain != 1.:
            log.debug("Multiplying {!s} by {!s}".format(name, self.gain))
        val = u * self.gain
        return sharedX(value=val, name=name)


class Constant(Initializer):
    """
    The :class:`Constant` an :class:`Initializer` that will create a variable of the constant repeating initial
    value or equal to the matrix of initial values.
    """
    def __init__(self, init_values=0):
        """
        Parameters
        ----------
        init_values : float or array_like
            Values to initialize as bias. If float, it repeats over `shape`. If array_like, initializes as the array.
        """
        self.init_values = init_values

    def __call__(self, shape, name=None):
        """
        Parameters
        ----------
        shape : tuple
            Tuple giving the shape information for the weight matrix.
        name : str
            Name to give the shared variable.

        Returns
        -------
        shared variable
            The shared variable matrix with given shape.
        """
        log.debug("Initializing bias %s variable with shape %s", name, str(shape))
        # init to zeros plus the offset
        val = as_floatX(numpy.ones(shape=shape, dtype=config.floatX) * self.init_values)
        return sharedX(value=val, name=name)
