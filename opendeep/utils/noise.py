"""
.. module:: noise
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
    Helper function to return a partially applied noise
    :param name: name of noise
    :type name: string
    :return: partial function
    :rtype: partial
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

def dropout(input, corruption_level=0.5, mrg=None, rescale=True):
    """
    This is the dropout function.
    :param input: tensor to apply dropout to
    :type input: tensor
    :param corruption_level: probability level for dropping an element (used in binomial distribution)
    :type corruption_level: float
    :param mrg: random number generator with a .binomial method
    :type mrg: random
    :param rescale: whether to rescale the output after dropout
    :type rescale: boolean
    :return: tensor with dropout applied
    :rtype: tensor
    """
    if mrg is None:
        mrg = theano_random

    keep_probability = 1 - corruption_level
    mask = mrg.binomial(p=keep_probability, n=1, size=input.shape, dtype=theano.config.floatX)

    output = (input * mask)

    if rescale:
        output = output / keep_probability

    return output

def add_gaussian(input, std=1, mrg=None):
    """
    This takes an input tensor and adds Gaussian noise to its elements with mean zero and provided standard deviation.
    :param input: tensor to add Gaussian noise to
    :type input: tensor
    :param std: standard deviation to use
    :type std: float
    :param mrg: random number generator with a .normal method
    :type mrg: random
    :return: tensor with Gaussian noise added
    :rtype: tensor
    """
    if mrg is None:
        mrg = theano_random
    log.debug('Adding Gaussian noise with std: %s', str(std))
    noise = mrg.normal(avg=0, std=std, size=input.shape, dtype=theano.config.floatX)
    OUT = input + noise
    return OUT

def add_uniform(input, interval, mrg=None):
    """
    This takes an intput tensor and adds noise drawn from a Uniform distribution from +- interval.
    :param input: tensor to add uniform noise to.
    :type input: tensor
    :param interval: range for noise to be drawn from (+- interval)
    :type interval: float
    :param mrg: random number generator with a .uniform method
    :type mrg: random
    :return: tensor with uniform noise added
    :rtype: tensor
    """
    if mrg is None:
        mrg = theano_random
    log.debug("Adding Uniform noise with interval [-%s, %s]", str(interval), str(interval))
    noise = mrg.uniform(low=-interval, high=interval, size=input.shape, dtype=theano.config.floatX)
    OUT = input + noise
    return OUT

def salt_and_pepper(input, corruption_level=0.2, mrg=None):
    """
    This applies salt and pepper noise to the input tensor - randomly setting bits to 1 or 0.
    :param input: the tensor to apply salt and pepper noise to
    :type input: tensor
    :param corruption_level: the amount of salt and pepper noise to add
    :type corruption_level: float
    :param mrg: random number generator with .binomial method
    :type mrg: random
    :return: tensor with salt and pepper noise applied
    :rtype: tensor
    """
    if mrg is None:
        mrg = theano_random
    # salt and pepper noise
    a = mrg.binomial(size=input.shape, n=1, p=(1 - corruption_level), dtype=theano.config.floatX)
    b = mrg.binomial(size=input.shape, n=1, p=0.5, dtype=theano.config.floatX)
    c = T.eq(a, 0) * b
    return input * a + c
