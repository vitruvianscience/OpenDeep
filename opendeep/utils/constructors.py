"""
This module provides functions for constructing Theano and Opendeep variables.
"""
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard imports
import logging
# third-party libraries
import numpy
import theano
import theano.tensor as T
from theano.compat.six import integer_types
# internal imports
from opendeep.utils.config import create_dictionary_like

log = logging.getLogger(__name__)

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
    # try:
    #     return sharedX(value=dataset, name=name, borrow=borrow, dtype=dtype)
    # except MemoryError:
    #     warnings.warn("Dataset was too big to fit in single shared variable, returning a tensor._shared instead...")
    #     return theano.tensor._shared(value=dataset, name=name, borrow=borrow)
    return theano.tensor._shared(value=dataset, name=name, borrow=borrow)

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