"""
.. module:: noise

This module provides the important noise functions - mostly used for regularization purposes to prevent the deep nets from
overfitting.

Based on code from Li Yao (University of Montreal)
https://github.com/yaoli/GSN
"""
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import logging
# third party libraries
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG

theano_random = RNG_MRG.MRG_RandomStreams(seed=23455)
# set a fixed number initializing RandomSate for 2 purpose:
#  1. repeatable experiments; 2. for multiple-GPU, the same initial weights

log = logging.getLogger(__name__)

def dropout(IN, corruption_level=0.5, MRG=None):
    """
    This is the dropout function.

    :param IN: tensor to apply dropout to
    :type IN: tensor

    :param corruption_level: probability level for dropping an element (used in binomial distribution)
    :type corruption_level: float

    :param MRG: random number generator with a .binomial method
    :type MRG: random

    :return: tensor with dropout applied
    :rtype: tensor
    """
    if MRG is None:
        MRG = theano_random

    mask = MRG.binomial(p=(1 - corruption_level), n=1, size=IN.shape)
    OUT = (IN * T.cast(mask, 'float32')) #/ cast32(corruption_level)
    return OUT

def add_gaussian(IN, std=1, MRG=None):
    """
    This takes an input tensor and adds Gaussian noise to its elements with mean zero and provided standard deviation.

    :param IN: tensor to add Gaussian noise to
    :type IN: tensor

    :param std: standard deviation to use
    :type std: float

    :param MRG: random number generator with a .normal method
    :type MRG: random

    :return: tensor with Gaussian noise added
    :rtype: tensor
    """
    if MRG is None:
        MRG = theano_random
    log.debug('GAUSSIAN NOISE : %s', str(std))
    noise = MRG.normal(avg=0, std=std, size=IN.shape, dtype=theano.config.floatX)
    OUT = IN + noise
    return OUT

def salt_and_pepper(IN, corruption_level=0.2, MRG=None):
    """
    This applies salt and pepper noise to the input tensor - randomly setting bits to 1 or 0.

    :param IN: the tensor to apply salt and pepper noise to
    :type IN: tensor

    :param corruption_level: the amount of salt and pepper noise to add
    :type corruption_level: float

    :param MRG: random number generator with .binomial method
    :type MRG: random

    :return: tensor with salt and pepper noise applied
    :rtype: tensor
    """
    if MRG is None:
        MRG = theano_random
    # salt and pepper noise
    a = MRG.binomial(size=IN.shape, n=1, p=(1 - corruption_level), dtype=theano.config.floatX)
    b = MRG.binomial(size=IN.shape, n=1, p=0.5, dtype=theano.config.floatX)
    c = T.eq(a, 0) * b
    return IN * a + c