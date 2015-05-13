"""
These functions provide the nonlinearities used as activation functions for the visible, hidden,
or output units in a deep net.
"""
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
# third party libraries
import theano.tensor as T
import theano.compat.six as six
# internal references
from opendeep import as_floatX

log = logging.getLogger(__name__)

def sigmoid(x):
    """
    See the Theano documentation.
    Returns the element-wise standard sigmoid nonlinearity applied to x

    Parameters
    ----------
    x : tensor
        Symbolic Tensor (or compatible).

    Returns
    -------
    tensor
        Element-wise sigmoid: sigmoid(x) = 1/(1+exp(-x)) applied to `x`

    .. note::

        You might want to try T.nnet.ultra_fast_sigmoid() or T.nnet.hard_sigmoid() for faster versions.
        Speed comparison for 100M float64 elements on a Core2 Duo @ 3.16 GHz:
        hard_sigmoid: 1.0s
        ultra_fast_sigmoid: 1.3s
        sigmoid (with amdlibm): 2.3s
        sigmoid (without amdlibm): 3.7s

        Precision: sigmoid(without or without amdlibm) > ultra_fast_sigmoid > hard_sigmoid.

    """
    # return T.nnet.hard_sigmoid(x)
    # return T.nnet.ultra_fast_sigmoid(x)
    return T.nnet.sigmoid(x)

def softmax(x):
    """
    See the Theano documentation.
    Returns the row-wise softmax function of x

    Parameters
    ----------
    x : 2D tensor
        Symbolic 2D Tensor (or compatible).

    Returns
    -------
    2D tensor
        Row-wise softmax: softmax_{ij}(x) = exp(x_{ij})/sum_k(exp(x_{ik})) applied to `x`.
    """
    return T.nnet.softmax(x)

def softplus(x):
    """
    See the Theano documentation.
    Returns the element-wise softplus nonlinearity applied to x.

    Parameters
    ----------
    x : tensor
        Symbolic tensor (or compatible).

    Returns
    -------
    tensor
        Element-wise softplus(x) = log_e (1 + exp(x)) applied to `x`.
    """
    return T.nnet.softplus(x)

def rectifier(x):
    """
    Returns the element-wise rectifier (ReLU) applied to x.

    Parameters
    ----------
    x : tensor
        Symbolic Tensor (or compatible).

    Returns
    -------
    tensor
        Element-wise rectifier: rectifier(x) = max(0,x) applied to `x`.

    .. note::

        This implementation uses rectifier(x) = (x + abs(x)) / 2
        which is faster than max(0,x)
        See https://github.com/SnippyHolloW/abnet/blob/807aeb9/layers.py#L15

    """
    # return T.maximum(as_floatX(0), x)
    # below fix is taken from Lasagne framework:
    # https://github.com/benanne/Lasagne/blob/master/lasagne/nonlinearities.py
    # The following is faster than lambda x: T.maximum(0, x)
    # Thanks to @SnippyHolloW for pointing this out.
    # See: https://github.com/SnippyHolloW/abnet/blob/807aeb9/layers.py#L15
    return (x + abs(x)) / as_floatX(2.0)

def tanh(x):
    """
    Returns the element-wise hyperbolic tangent (tanh) applied to x.

    Parameters
    ----------
    x : tensor
        Symbolic Tensor (or compatible).

    Returns
    -------
    tensor
        Element-wise tanh: tanh(x) = (1 - exp(-2x))/(1 + exp(-2x)) applied to `x`.
    """
    return T.tanh(x)

def linear(x):
    """
    Returns the linear function of x, which is just x. This method effectively does nothing, but counts as a name for
    readability elsewhere when constructing layers.

    Parameters
    ----------
    x : object
        Input to return the identity of.

    Returns
    -------
    object
        Returns `x` without altering.
    """
    return x

############# keep activation functions above this line, and add them to the dictionary below #############
# this is a dictionary containing a string keyword mapping to the activation function -
# used for get_activation_function(name)
_activations = {'sigmoid': sigmoid,
                'softmax': softmax,
                'softplus': softplus,
                'rectifier': rectifier,
                'relu': rectifier,  # shorter alternative name for rectifier
                'tanh': tanh,
                'linear': linear}

def is_binary(activation):
    """
    returns if the activation function is binary

    Parameters
    ----------
    activation : function
        The activation function to see if it is binary.

    Returns
    -------
    bool
        Boolean if it is a binary function (within range [0,1] (default to False).
    """
    binary = False
    if activation == sigmoid or activation == softmax:
        binary = True

    return binary

def get_activation_function(name):
    """
    This helper method returns the appropriate activation function given a string name. It looks up the appropriate
    function from the internal _activations dictionary.

    Parameters
    ----------
    name : str or Callable
        String representation of the function you want (see options in the _activations dictionary).
        Or, it could already be a function (Callable).

    Returns
    -------
    function
        The appropriate activation function.

    Raises
    ------
    NotImplementedError
        If the function was not found in the dictionary.
    """
    # if the activation is None, return identity function (no activation)
    if name is None:
        return linear
    # return the function itself if it is a Callable
    elif callable(name):
        return name
    # otherwise if it is a string
    elif isinstance(name, six.string_types):
        # standardize the input to be lowercase
        name = name.lower()
        # grab the appropriate activation function from the dictionary of activations
        func = _activations.get(name)
        # if it couldn't find the function (key didn't exist), raise a NotImplementedError
        if func is None:
            log.critical("Did not recognize activation %s! Please use one of: ", str(name), str(_activations.keys()))
            raise NotImplementedError(
                "Did not recognize activation {0!s}! Please use one of: {1!s}".format(name, _activations.keys())
            )
        # return the found function
        return func
    # else we don't know what to do so throw error
    else:
        log.critical("Activation function not implemented for %s with type %s", str(name), str(type(name)))
        raise NotImplementedError("Activation function not implemented for %s with type %s", str(name), str(type(name)))