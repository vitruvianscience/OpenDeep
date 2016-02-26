"""
Provides various methods for initializing shared weights and bias variables.

Based on code from Li Yao (University of Montreal)
https://github.com/yaoli/GSN

And theano_alexnet (https://github.com/uoguelph-mlrg/theano_alexnet)
"""
# standard libraries
import logging
from functools import partial
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

# these are the possible formulas for the interval when building weights from a uniform distribution
_uniform_interval = {
    # shape[0] = rows
    # shape[1] = cols
    # numpy.prod(shape[2:]) = receptive field from Glorot et al.
    # use this only when the activation function is sigmoid
    'sigmoid': lambda shape: 4 * numpy.sqrt(6. / ((shape[0] + shape[1]) * numpy.prod(shape[2:]))),
    # this is the default provided in other codebases
    'default': lambda shape: 1 / numpy.sqrt(shape[0]),
    # this is the default for the GSN code from Li Yao
    'glorot': lambda shape: numpy.sqrt(6. / ((shape[0] + shape[1]) * numpy.prod(shape[2:]))),
    'montreal': lambda shape: numpy.sqrt(6. / ((shape[0] + shape[1]) * numpy.prod(shape[2:])))  # compatibility syntax
}

def get_weights(weights_init, shape,
                gain=1., mean=None, std=None, interval=None, add_noise=None, rng=None, name="W", **kwargs):
    """
    This will initialize the weights from the method passed from weights_init with the arguments in kwargs.

    Parameters
    ----------
    weights_init : str
        String name of the method for creating weights. Can be 'gaussian', 'uniform', 'identity', or 'orthogonal'
    shape : tuple
        Tuple of the shape you want the weight matrix.
    gain : float or str
        A multiplicative factor to affect the whole weights matrix. If gain is the string 'rectifier' or 'relu', it
        will default to a value that is normally good for those activations (sqrt(2)).
    mean : float
        Mean value for using gaussian weights.
    std : float
        Standard deviation for using gaussian weights
    interval : float or str
        +- interval to use for uniform weights. If a string, it will look up the appropriate method in the
        _uniform_interval dictionary.
    rng : random
        Theano or numpy random number generator to use for sampling.
    name : str
        Name for the returned tensor shared variable.

    Returns
    -------
    shared variable
        Theano tensor (shared variable) for the weights.
    """
    # check if the gain is the preset for relu activation
    if isinstance(gain, string_types):
        if gain.lower() == 'relu' or gain.lower() == 'rectifier':
            gain = numpy.sqrt(2)
        else:
            log.error("Did not recognize gain %s. Needs to be 'rectifier'. Defaulting to 1." % gain)
            gain = 1.
    elif gain is None:
        gain = 1.

    # make sure the weights_init is a string to the method to use
    if isinstance(weights_init, string_types):
        weights_init = weights_init.lower()
        # if we are initializing weights from a normal distribution
        if weights_init == 'gaussian':
            return get_weights_gaussian(shape=shape, mean=mean, std=std, name=name, rng=rng, gain=gain)
        # if we are initializing weights from a uniform distribution
        elif weights_init == 'uniform' or weights_init in _uniform_interval:
            if weights_init in _uniform_interval:
                interval = weights_init
            return get_weights_uniform(shape=shape, interval=interval, name=name, rng=rng, gain=gain)
        # if we are initializing an identity matrix
        elif weights_init == 'identity':
            return get_weights_identity(shape=shape, name=name, add_noise=add_noise, gain=gain)
        # if we are initializing an orthonormal matrix
        elif weights_init == 'orthogonal' or weights_init == 'ortho':
            return get_weights_orthogonal(shape=shape, name=name, rng=rng, gain=gain)

    # otherwise not implemented
    log.error("Did not recognize weights_init %s! Pleas try gaussian, uniform, identity, or orthogonal."
              % str(weights_init))
    raise NotImplementedError("Did not recognize weights_init %s! Pleas try gaussian, uniform, identity, or orthogonal"
                              % str(weights_init))

def get_weights_uniform(shape, interval='glorot', name="W", rng=None, gain=1.):
    """
    This initializes a shared variable with a given shape for weights drawn from a Uniform distribution with
    low = -interval and high = interval.

    Interval can either be a number to use, or a string key to one of the predefined formulas in the
    _uniform_interval dictionary.

    Parameters
    ----------
    shape : tuple
        A tuple giving the shape information for this weight matrix.
    interval : float or str
        Either a number for your own custom interval, or a string key to one of the predefined formulas.
    name : str
        The name to give the shared variable.
    rng : random
        The random number generator to use with a .uniform method.
    gain : float
        A multiplicative factor to affect the whole weights matrix.

    Returns
    -------
    shared variable
        The theano shared variable with given shape and name drawn from a uniform distribution.

    Raises
    ------
    NotImplementedError
        If the string name for the interval couldn't be found in the dictionary.
    """
    if rng is None:
        rng = numpy.random
    # If the interval parameter is a string, grab the appropriate formula from the function dictionary,
    # and apply the appropriate shape numbers to it.
    if isinstance(interval, string_types):
        interval_func = _uniform_interval.get(interval)
        if interval_func is None:
            log.error('Could not find uniform interval formula %s, try one of %s instead.' %
                      (str(interval), str(_uniform_interval.keys())))
            raise NotImplementedError('Could not find uniform interval formula %s, try one of %s instead.' %
                                      (str(interval), str(_uniform_interval.keys())))
        else:
            log.debug("Creating weights %s with shape %s from Uniform distribution with formula name: %s",
                      name, str(shape), str(interval))
            interval = interval_func(shape)
    else:
        log.debug("Creating weights %s with shape %s from Uniform distribution with given interval +- %s",
                  name, str(shape), str(interval))
    # build the uniform weights tensor
    val = as_floatX(rng.uniform(low=-interval, high=interval, size=shape))
    # check if a theano rng was used
    if isinstance(val, TensorVariable):
        val = val.eval()

    val = val * gain
    # make it into a shared variable
    return sharedX(value=val, name=name)

def get_weights_gaussian(shape, mean=None, std=None, name="W", rng=None, gain=1.):
    """
    This initializes a shared variable with the given shape for weights drawn from a
    Gaussian distribution with mean and std.

    Parameters
    ----------
    shape : tuple
        A tuple giving the shape information for this weight matrix.
    mean : float
        The mean to use for the Gaussian distribution.
    std : float
        The standard deviation to use dor the Gaussian distribution.
    name : str
        The name to give the shared variable.
    rng : random
        A given random number generator to use with .normal method.
    gain : float
        A multiplicative factor to affect the whole weights matrix.

    Returns
    -------
    shared variable
        The theano shared variable with given shape and drawn from a Gaussian distribution.
    """
    default_mean = 0
    default_std  = 0.05

    mean = mean or default_mean
    std = std or default_std

    log.debug("Creating weights %s with shape %s from Gaussian mean=%s, std=%s", name, str(shape), str(mean), str(std))
    if rng is None:
        rng = numpy.random

    if std != 0:
        if isinstance(rng, type(numpy.random)):
            val = numpy.asarray(rng.normal(loc=mean, scale=std, size=shape), dtype=config.floatX)
        else:
            val = numpy.asarray(rng.normal(avg=mean, std=std, size=shape).eval(), dtype=config.floatX)
    else:
        val = as_floatX(mean * numpy.ones(shape, dtype=config.floatX))

    # check if a theano rng was used
    if isinstance(val, TensorVariable):
        val = val.eval()

    val = val * gain
    # make it into a shared variable
    return sharedX(value=val, name=name)

def get_weights_identity(shape, name="W", add_noise=None, gain=1.):
    """
    This will return a weights matrix as close to the identity as possible. If a non-square shape, it will make
    a matrix of the form (I 0)

    Identity matrix for weights is useful for RNNs with ReLU! http://arxiv.org/abs/1504.00941

    Parameters
    ----------
    shape : tuple
        Tuple giving the shape information for the weight matrix.
    name : str
        Name to give the shared variable.
    add_noise : functools.partial
        A partially applied noise function (just missing the input parameter) to add noise to the identity
        initialization. Noise functions can be found in opendeep.utils.noise.
    gain : float
        A multiplicative factor to affect the whole weights matrix.

    Returns
    -------
    shared variable
        The theano shared variable identity matrix with given shape.
    """
    log.debug("Creating Identity matrix weights %s with shape %s", name, str(shape))
    weights = numpy.eye(N=shape[0], M=int(numpy.prod(shape[1:])), k=0, dtype=config.floatX)

    if add_noise:
        if isinstance(add_noise, partial):
            weights = add_noise(input=weights)
        else:
            log.error("Add noise to identity weights was not a functools.partial object. Ignoring...")

    val = weights * gain
    return sharedX(value=val, name=name)

def get_weights_orthogonal(shape, name="W", rng=None, gain=1.):
    """
    This returns orthonormal random values to initialize a weight matrix (using SVD).

    Some discussion here:
    http://www.reddit.com/r/MachineLearning/comments/2qsje7/how_do_you_initialize_your_neural_network_weights/

    From Lasagne:
    For n-dimensional shapes where n > 2, the n-1 trailing axes are flattened.
    For convolutional layers, this corresponds to the fan-in, so this makes the initialization
    usable for both dense and convolutional layers.

    Parameters
    ----------
    shape : tuple
        Tuple giving the shape information for the weight matrix.
    name : str
        Name to give the shared variable.
    rng : random
        A given random number generator to use with .normal method.
    gain : float
        A multiplicative factor to affect the whole weights matrix.

    Returns
    -------
    shared variable
        The theano shared variable orthogonal matrix with given shape.
    """
    log.debug("Creating Orthogonal matrix weights %s with shape %s", name, str(shape))
    if rng is None:
        rng = numpy.random

    if len(shape) == 1:
        shape = (shape[0], shape[0])
    else:
        # flatten shapes bigger than 2
        # From Lasagne: For n-dimensional shapes where n > 2, the n-1 trailing axes are flattened.
        # For convolutional layers, this corresponds to the fan-in, so this makes the initialization
        # usable for both dense and convolutional layers.
        shape = (shape[0], numpy.prod(shape[1:]))

    # Sample from the standard normal distribution
    if isinstance(rng, type(numpy.random)):
        a = numpy.asarray(rng.normal(loc=0., scale=1., size=shape), dtype=config.floatX)
    else:
        a = numpy.asarray(rng.normal(avg=0., std=1., size=shape).eval(), dtype=config.floatX)

    u, _, _ = numpy.linalg.svd(a, full_matrices=False)

    val = u * gain
    return sharedX(value=val, name=name)

def get_bias(shape, name="b", init_values=None):
    """
    This creates a theano shared variable for the bias parameter - normally initialized to zeros,
    but you can specify other values

    Parameters
    ----------
    shape : tuple
        The shape to use for the bias vector/matrix.
    name : str
        The name to give the shared variable.
    offset : float or array_like
        Values to add to the zeros, if you want a nonzero bias initially.

    Returns
    -------
    shared variable
        The theano shared variable with given shape.
    """
    default_init = 0

    init_values = init_values or default_init

    log.debug("Initializing bias %s variable with shape %s", name, str(shape))
    # init to zeros plus the offset
    val = as_floatX(numpy.ones(shape=shape, dtype=config.floatX) * init_values)
    return sharedX(value=val, name=name)
