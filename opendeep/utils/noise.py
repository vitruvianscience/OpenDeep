"""
This module provides the important noise functions - mostly used for regularization purposes to prevent the
deep nets from overfitting.

Based on code from Li Yao (University of Montreal)
https://github.com/yaoli/GSN
"""
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
from functools import partial
# third party libraries
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG
import theano.compat.six as six

theano_random = RNG_MRG.MRG_RandomStreams(seed=23455)
# set a fixed number initializing RandomSate for 2 purpose:
#  1. repeatable experiments; 2. for multiple-GPU, the same initial weights

log = logging.getLogger(__name__)

def get_noise(name, *args, **kwargs):
    """
    Helper function to return a partially applied noise functions - all you need to do is apply them to an input.

    Parameters
    ----------
    name : str
        Name of noise function to use (key in a function dictionary).

    Returns
    -------
    partial
        Partially applied function with the input arguments.
    """
    noise_lookup = {
        'dropout': dropout,
        'gaussian': add_gaussian,
        'uniform': add_uniform,
        'salt_and_pepper': salt_and_pepper
    }
    if isinstance(name, six.string_types):
        if name in noise_lookup:
            return partial(noise_lookup[name], *args, **kwargs)
        else:
            log.error("Couldn't find noise %s, try one of %s.", name, str(noise_lookup.keys()))
            raise AssertionError("Couldn't find noise %s, try one of %s." % (name, str(noise_lookup.keys())))
    else:
        log.error("Noise name needs to be a string, found %s", str(name))
        raise AssertionError("Noise name needs to be a string, found %s" % str(name))

def dropout(input, noise_level=0.5, mrg=None, rescale=True):
    """
    This is the dropout function.

    Parameters
    ----------
    input : tensor
        Tensor to apply dropout to.
    corruption_level : float
        Probability level for dropping an element (used in binomial distribution).
    mrg : random
        Random number generator with a .binomial method.
    rescale : bool
        Whether to rescale the output after dropout.

    Returns
    -------
    tensor
        Tensor with dropout applied.
    """
    if mrg is None:
        mrg = theano_random

    keep_probability = 1 - noise_level
    mask = mrg.binomial(p=keep_probability, n=1, size=input.shape, dtype=theano.config.floatX)

    output = (input * mask)

    if rescale:
        output = output / keep_probability

    return output

def add_gaussian(input, noise_level=1, mrg=None):
    """
    This takes an input tensor and adds Gaussian noise to its elements with mean zero and provided standard deviation.

    Parameters
    ----------
    input : tensor
        Tensor to add Gaussian noise to.
    noise_level : float
        Standard deviation to use.
    mrg : random
        Random number generator with a .normal method.

    Returns
    -------
    tensor
        Tensor with Gaussian noise added.
    """
    if mrg is None:
        mrg = theano_random
    log.debug('Adding Gaussian noise with std: %s', str(noise_level))
    noise = mrg.normal(avg=0, std=noise_level, size=input.shape, dtype=theano.config.floatX)
    OUT = input + noise
    return OUT

def add_uniform(input, noise_level, mrg=None):
    """
    This takes an intput tensor and adds noise drawn from a Uniform distribution from +- interval.

    Parameters
    ----------
    input : tensor
        Tensor to add uniform noise to.
    noise_level : float
        Range for noise to be drawn from (+- interval).
    mrg : random
        Random number generator with a .uniform method.

    Returns
    -------
    tensor
        Tensor with uniform noise added.
    """
    if mrg is None:
        mrg = theano_random
    log.debug("Adding Uniform noise with interval [-%s, %s]", str(noise_level), str(noise_level))
    noise = mrg.uniform(low=-noise_level, high=noise_level, size=input.shape, dtype=theano.config.floatX)
    OUT = input + noise
    return OUT

def salt_and_pepper(input, noise_level=0.2, mrg=None):
    """
    This applies salt and pepper noise to the input tensor - randomly setting bits to 1 or 0.

    Parameters
    ----------
    input : tensor
        The tensor to apply salt and pepper noise to.
    noise_level : float
        The amount of salt and pepper noise to add.
    mrg : random
        Random number generator with .binomial method.

    Returns
    -------
    tensor
        Tensor with salt and pepper noise applied.
    """
    if mrg is None:
        mrg = theano_random
    # salt and pepper noise
    a = mrg.binomial(size=input.shape, n=1, p=(1 - noise_level), dtype=theano.config.floatX)
    b = mrg.binomial(size=input.shape, n=1, p=0.5, dtype=theano.config.floatX)
    c = T.eq(a, 0) * b
    return input * a + c
