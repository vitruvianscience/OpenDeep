"""
These functions provide the nonlinearities used as activation functions for the visible, hidden,
or output units in a deep net.
"""
# standard libraries
import logging
from functools import partial
# third party libraries
import theano
import theano.tensor as T
from six import string_types

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
    Returns the row-wise softmax function of x.

    In the case of 3D input, it returns the scan of softmax applied over the second two dimensions
    (loops over first dimension).

    Parameters
    ----------
    x : 2D or 3D tensor
        Symbolic 2D or 3D Tensor (or compatible).

    Returns
    -------
    2D or 3D tensor
        Row-wise softmax: softmax_{ij}(x) = exp(x_{ij})/sum_k(exp(x_{ik})) applied to `x`. Returns same shape as input.
    """
    if x.ndim == 3:
        cost, _ = theano.scan(fn=T.nnet.softmax, sequences=x)
        return cost
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

def rectifier(x, leaky=0):
    """
    Returns the element-wise rectifier (ReLU) applied to x.

    Parameters
    ----------
    x : tensor
        Symbolic Tensor (or compatible).
    leaky: scalar or tensor
        Slope for negative input, usually between 0 and 1. The default value of 0 will lead to the standard rectifier,
        1 will lead to a linear activation function, and any value in between will give a leaky rectifier.
        A shared variable (broadcastable against x) will result in a parameterized rectifier with learnable slope(s).

    Returns
    -------
    tensor
        Element-wise rectifier: rectifier(x) = max(0,x) applied to `x`.

    """
    # return T.maximum(as_floatX(0), x)
    # below fix is taken from Lasagne framework:
    # https://github.com/benanne/Lasagne/blob/master/lasagne/nonlinearities.py
    # The following is faster than lambda x: T.maximum(0, x)
    # Thanks to @SnippyHolloW for pointing this out.
    # See: https://github.com/SnippyHolloW/abnet/blob/807aeb9/layers.py#L15
    # return (x + abs(x)) / as_floatX(2.0)
    return T.nnet.relu(x, alpha=leaky)

def elu(x, alpha=1):
    """
    (from Lasagne https://github.com/Lasagne/Lasagne/blob/master/lasagne/nonlinearities.py)

    Exponential Linear Unit :math:`\\varphi(x) = (x > 0) ? x : e^x - 1`
    The Exponential Linear Unit (EUL) was introduced in [1]_. Compared to the
    linear rectifier :func:`rectify`, it has a mean activation closer to zero
    and nonzero gradient for negative input, which can help convergence.
    Compared to the leaky rectifier, it saturates for
    highly negative inputs.

    Parameters
    ----------
    x : float32
        The activation (the summed, weighed input of a neuron).
    Returns
    -------
    float32
        The output of the exponential linear unit for the activation.

    References
    ----------
    .. [1] Djork-Arne Clevert, Thomas Unterthiner, Sepp Hochreiter (2015):
       Fast and Accurate Deep Network Learning by Exponential Linear Units
       (ELUs), http://arxiv.org/abs/1511.07289
    """
    assert alpha > 0, "alpha parameter to ELU has to be > 0."
    return T.switch(x > 0, x, alpha*(T.exp(x) - 1))

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
                'linear': linear,
                'identity': linear,
                'elu': elu}

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

def get_activation_function(name, *args, **kwargs):
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
    elif isinstance(name, string_types):
        # standardize the input to be lowercase
        name = name.lower()
        # grab the appropriate activation function from the dictionary of activations
        if name in _activations:
            func = partial(_activations[name], *args, **kwargs)
        else:
            # if it couldn't find the function (key didn't exist), raise a NotImplementedError
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