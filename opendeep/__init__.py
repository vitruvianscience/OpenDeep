"""
Code used throughout the entire OpenDeep package.
"""
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

import version
__version__ = version.__version__

# third-party libraries
import theano
from theano.compat.six import integer_types
import numpy
# internal imports
from opendeep.utils.config import create_dictionary_like
from opendeep.utils.decorators import *

def trunc(input, length=8):
    """
    Casts the input to a string and cuts it off after `length` characters.

    Parameters
    ----------
    input : object
        The input to truncate. Must be able to convert to String.
    length : int, optional
        The length of the resulting string (number of characters).

    Returns
    -------
    str
        The appropriately truncated string representation of `input`.
    """
    return str(input)[:length]

def logit(p):
    """
    The logit function of a probability p. This represents the inverse of the sigmoidal function.

    .. math::
        logit(p) = \\log{\\left(\\frac{p}{1-p}\\right)}

    Parameters
    ----------
    p : float
        A probability value to compute the logit.

    Returns
    -------
    float
        The result of the logit function.
    """
    return numpy.log(p / (1 - p))

def binarize(input, cutoff=0.5):
    """
    Elementwise converts the input to 0 or 1.
    If element >= `cutoff` : 1; otherwise : 0.

    Parameters
    ----------
    input : tensor or array
        The number, vector, matrix, or tensor to binarize.
    cutoff : float
        The threshold value between [0, 1].

    Returns
    -------
    tensor or numpy array
        The input converted to 0 or 1 and cast to float.
    """
    return as_floatX(input >= cutoff)

def sigmoid(x):
    """
    The elementwise sigmoid function applied to `x`.

    .. math:: sigmoid(x) = \\frac{1}{1 + e^{-x}}

    Parameters
    ----------
    x : number, vector, matrix, or tensor
        The input to perform the sigmoid function on.

    Returns
    -------
    tensor or numpy array
        The input `x` put through the elementwise sigmoid and cast to float.
    """
    return as_floatX(1. / (1 + numpy.exp(-x)))

def function(*args, **kwargs):
    """
    A wrapper around theano.function that disables the on_unused_input error (sets `on_unused_input` = "warn").
    Almost no part of OpenDeep can assume that an unused input is an error, so
    the default from Theano is inappropriate for this project.

    See: http://deeplearning.net/software/theano/library/compile/function.html

    Parameters
    ----------
    *args
        Variable length argument list to theano.function.
    **kwargs
        Arbitrary keyword arguments to theano.function.

    Returns
    -------
    theano.function
        Compiled Theano function.
    """
    return theano.function(*args, on_unused_input='warn', **kwargs)

def grad(*args, **kwargs):
    """
    A wrapper around theano.gradient.grad that disable the disconnected_inputs
    error (sets `disconnected_inputs` = "warn"). Almost no part of OpenDeep can assume that a disconnected input
    is an error.

    See: http://deeplearning.net/software/theano/library/tensor/basic.html#theano.gradient.grad

    Parameters
    ----------
    *args
        Variable length argument list to theano.gradient.grad.
    **kwargs
        Arbitrary keyword arguments to theano.gradient.grad.

    Returns
    -------
    variable or list/tuple of Variables (matching `wrt`)
        Symbolic expression of gradient of `cost` with respect to each of the `wrt` terms.
        If an element of `wrt` is not differentiable with respect to the output, then a zero variable is returned.
        It returns an object of same type as `wrt`: a list/tuple or Variable in all cases.
    """
    return theano.gradient.grad(*args, disconnected_inputs='warn', **kwargs)

def sharedX(value, name=None, borrow=False, dtype=theano.config.floatX):
    """
    Transform value into a theano shared variable of type `dtype` or theano.config.floatX

    For Theano shared variables, see: http://deeplearning.net/software/theano/library/compile/shared.html

    Parameters
    ----------
    value : number, array, vector, matrix, or tensor
        The input value to create into a Theano shared variable.
    name : string, optional
        The name for this shared variable.
    borrow : bool, optional
        The boolean `borrow` value to use in the Theano `shared` function.
    dtype : string, optional
        The `dtype` to use during conversion. Defaults to theano.config.floatX.

    Returns
    -------
    SharedVariable
        The Theano shared variable of the input `value`.
    """
    return theano.shared(theano._asarray(value, dtype=dtype),
                         name=name,
                         borrow=borrow)

def dataset_shared(dataset, name=None, borrow=False, dtype=theano.config.floatX):
    """
    Transform input `dataset` into a Theano shared variable of type `dtype`.

    .. todo:: Currently acts as a wrapper for `sharedX`. This is used for datasets,
        so we might want to use theano.tensor._shared instead of theano.shared for GPU optimizations.

    Parameters
    ----------
    dataset : number, array, vector, matrix, or tensor
        The input dataset to create into a Theano shared variable.
    name : string, optional
        The name for this shared variable.
    borrow : bool, optional
        The boolean `borrow` value to use in the Theano `shared` function.
    dtype : string, optional
        The `dtype` to use during conversion. Defaults to theano.config.floatX.

    Returns
    -------
    SharedVariable
        The Theano shared variable of the input `dataset`.
    """
    # TODO: look into using theano.tensor._shared instead? So it can bring portions to the GPU as needed?
    return sharedX(value=dataset, name=name, borrow=borrow, dtype=dtype)

def as_floatX(variable):
    """
    Casts a given variable into dtype `theano.config.floatX`. Numpy ndarrays will
    remain numpy ndarrays, python floats will become 0-D ndarrays and
    all other types will be treated as theano tensors.

    Parameters
    ----------
    variable: int, float, numpy array, or tensor
        The input to convert to type `theano.config.floatX`.

    Returns
    -------
    numpy array or tensor
        The input `variable` casted as type `theano.config.floatX`.
    """
    if isinstance(variable, (integer_types, float, numpy.number, numpy.ndarray)):
        return numpy.cast[theano.config.floatX](variable)

    return theano.tensor.cast(variable, theano.config.floatX)

def constantX(value):
    """
    Returns a constant of value `value` with type theano.config.floatX.

    Parameters
    ----------
    value : array_like
        The input value to create a Theano constant.

    Returns
    -------
    Constant
        A theano.tensor.constant of the input `value`.
    """
    return theano.tensor.constant(
        numpy.asarray(value, dtype=theano.config.floatX)
    )

def safe_zip(*args):
    """
    Like zip, but ensures arguments are of same length.

    Parameters
    ----------
    *args
        Argument list to `zip`

    Returns
    -------
    list
        The zipped list of inputs.

    Raises
    ------
    ValueError
        If the length of any argument is different than the length of args[0].
    """
    base = len(args[0])
    for i, arg in enumerate(args[1:]):
        if len(arg) != base:
            raise ValueError("Argument[0] has length %d but argument %d has "
                             "length %d" % (base, i+1, len(arg)))
    return zip(*args)

def init_from_config(class_type, config):
    """
    Takes a class type and a configuration, and instantiates the class with the parameters in config.

    Parameters
    ----------
    class_type : Class
        The class type to create a new instance of.
    config : a json, yaml, or dictionary-like object
        The configuration to instantiate a new class of `class_type` with.

    Returns
    -------
    Class
        A newly instantiated class with parameters from `config`.
    """
    config_dict = create_dictionary_like(config)
    if config_dict is None:
        config_dict = {}
    return class_type(**config_dict)