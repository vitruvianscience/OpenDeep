"""
.. module:: nnet

Based on code from Li Yao (University of Montreal)
https://github.com/yaoli/GSN

and theano_alexnet (https://github.com/uoguelph-mlrg/theano_alexnet)
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
import numpy
import theano
import theano.tensor as T
# internal imports
from opendeep import cast_floatX

log = logging.getLogger(__name__)

numpy.random.RandomState(23455)
# set a fixed number initializing RandomSate for 2 purpose:
#  1. repeatable experiments; 2. for multiple-GPU, the same initial weights

# these are the possible formulas for the interval when building weights from a uniform distribution
_uniform_interval = {
    # shape[0] = rows
    # shape[1] = cols
    # numpy.prod(shape[2:]) = receptive field from Glorot et al.
    'sigmoid': lambda shape: 4 * numpy.sqrt(6. / ((shape[0] + shape[1]) * numpy.prod(shape[2:]))),  # use this only when the activation function is sigmoid
    'default': lambda shape: 1 / numpy.sqrt(shape[0]),  # this is the default provided in other codebases
    'montreal': lambda shape: numpy.sqrt(6. / ((shape[0] + shape[1]) * numpy.prod(shape[2:])))  # this is the default for the GSN code from Li Yao
}
default_interval = 'montreal'

def get_weights_uniform(shape, interval=None, name="W", rng=None):
    """
    This initializes a shared variable with a given shape for weights drawn from a Uniform distribution with
    low = -interval and high = interval.

    Interval can either be a number to use, or a string key to one of the predefined formulas in the _uniform_interval dictionary.

    :param shape: a tuple giving the shape information for this weight matrix
    :type shape: Tuple

    :param interval: either a number for your own custom interval, or a string key to one of the predefined formulas
    :type interval: Float or String

    :param name: the name to give the shared variable
    :type name: String

    :param rng: the random number generator to use with a .uniform method
    :type rng: random

    :return: the theano shared variable with given shape and name drawn from a uniform distribution
    :rtype: shared variable

    :raises: NotImplementedError
    """
    interval = interval or default_interval

    if rng is None:
        rng = numpy.random
    # If the interval parameter is a string, grab the appropriate formula from the function dictionary, and apply the appropriate
    # shape numbers to it.
    if isinstance(interval, basestring):
        interval_func = _uniform_interval.get(interval)
        if interval_func is None:
            log.error('Could not find uniform interval formula %s, try one of %s instead.' %
                      str(interval), str(_uniform_interval.keys()))
            raise NotImplementedError('Could not find uniform interval formula %s, try one of %s instead.' %
                                      str(interval), str(_uniform_interval.keys()))
        else:
            log.debug("Creating weights with shape %s from Uniform distribution with formula name: %s", str(shape), str(interval))
            interval = interval_func(shape)
    else:
        log.debug("Creating weights with shape %s from Uniform distribution with given interval +- %s", str(shape), str(interval))
    # build the uniform weights tensor
    val = cast_floatX(rng.uniform(low=-interval, high=interval, size=shape))
    return theano.shared(value=val, name=name)

def get_weights_gaussian(shape, mean=None, std=None, name="W", rng=None):
    """
    This initializes a shared variable with the given shape for weights drawn from a Gaussian distribution with mean and std.

    :param shape: a tuple giving the shape information for this weight matrix
    :type shape: Tuple

    :param mean: the mean to use for the Gaussian distribution
    :type mean: float

    :param std: the standard deviation to use dor the Gaussian distribution
    :type std: float

    :param name: the name to give the shared variable
    :type name: String

    :param rng: a given random number generator to use with .normal method
    :type rng: random

    :return: the theano shared variable with given shape and drawn from a Gaussian distribution
    :rtype: shared variable
    """
    default_mean = 0
    default_std  = 0.05

    mean = mean or default_mean
    std = std or default_std

    log.debug("Creating weights with shape %s from Gaussian mean=%s, std=%s", str(shape), str(mean), str(std))
    if rng is None:
        rng = numpy.random

    if std != 0:
        val = numpy.asarray(rng.normal(loc=mean, scale=std, size=shape), dtype=theano.config.floatX)
    else:
        val = cast_floatX(mean * numpy.ones(shape, dtype=theano.config.floatX))

    return theano.shared(value=val, name=name)

def get_bias(shape, name="b", init_values=None):
    """
    This creates a theano shared variable for the bias parameter - normally initialized to zeros, but you can specify other values

    :param shape: the shape to use for the bias vector/matrix
    :type shape: Tuple

    :param name: the name to give the shared variable
    :type name: String

    :param offset: values to add to the zeros, if you want a nonzero bias initially
    :type offset: float/vector

    :return: the theano shared variable with given shape
    :rtype: shared variable
    """
    default_init = 0

    init_values = init_values or default_init

    log.debug("Initializing bias variable with shape %s" % str(shape))
    # init to zeros plus the offset
    val = cast_floatX(numpy.ones(shape=shape, dtype=theano.config.floatX) * init_values)
    return theano.shared(value=val, name=name)

def mirror_images(input, image_shape, cropsize, rand, flag_rand):
    """
    This takes an input batch of images (normally the input to a convolutional net), and augments them by mirroring and concatenating.

    :param input: the input 4D tensor of images
    :type input: Tensor4D

    :param image_shape: the shape of the 4D tensor input
    :type image_shape: Tuple

    :param cropsize: what size to crop to
    :type cropsize: Integer

    :param rand: a vector representing a random array for cropping/mirroring the data
    :type rand: fvector

    :param flag_rand: to randomize the mirror
    :type flag_rand: Boolean

    :return: tensor4D representing the mirrored/concatenated input
    :rtype: same as input
    """
    # The random mirroring and cropping in this function is done for the
    # whole batch.

    # trick for random mirroring
    mirror = input[:, :, ::-1, :]
    input = T.concatenate([input, mirror], axis=0)

    # crop images
    center_margin = (image_shape[2] - cropsize) / 2

    if flag_rand:
        mirror_rand = T.cast(rand[2], 'int32')
        crop_xs = T.cast(rand[0] * center_margin * 2, 'int32')
        crop_ys = T.cast(rand[1] * center_margin * 2, 'int32')
    else:
        mirror_rand = 0
        crop_xs = center_margin
        crop_ys = center_margin

    output = input[mirror_rand * 3:(mirror_rand + 1) * 3, :, :, :]
    output = output[:, crop_xs:crop_xs + cropsize, crop_ys:crop_ys + cropsize, :]

    log.debug("mirrored input data with shape_in: " + str(image_shape))

    return output

def bc01_to_c01b(input):
    """
    This helper method uses dimshuffle on a 4D input tensor assumed to be bc01 format (batch, channels, rows, cols) and outputs it
    in c01b format (channels, rows, cols, batch). This operation is used for convolutions.

    :param input: a 4D input tensor assumed to be in bc01 ordering
    :type input: 4D tensor

    :return: a 4D tensor in c01b ordering
    :rtype: 4D tensor
    """
    # make sure it is a 4d tensor:
    assert input.ndim == 4
    # return the shuffle
    return input.dimshuffle(1, 2, 3, 0)

def c01b_to_bc01(input):
    """
    This helper method uses dimshuffle on a 4D input tensor assumed to be c01b format (channels, rows, cols, batch) and outputs it
    in bc01 format (batch, channels, rows, cols). This operation is used for convolutions.

    :param input: a 4D input tensor assumed to be in c01b ordering
    :type input: 4D tensor

    :return: a 4D tensor in bc01 ordering
    :rtype: 4D tensor
    """
    # make sure it is a 4d tensor:
    assert input.ndim == 4
    # return the shuffle
    return input.dimshuffle(3, 0, 1, 2)

def cross_channel_normalization_bc01(bc01, alpha=1e-4, k=2, beta=0.75, n=5):
    """
    BC01 format (batch, channels, rows, cols) version of Cross Channel Normalization.

    See "ImageNet Classification with Deep Convolutional Neural Networks"
    Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton
    NIPS 2012
    Section 3.3, Local Response Normalization

    f(c01b)_[i,j,k,l] = c01b[i,j,k,l] / scale[i,j,k,l]
    scale[i,j,k,l] = (k + sqr(c01b)[clip(i-n/2):clip(i+n/2),j,k,l].sum())^beta
    clip(i) = T.clip(i, 0, c01b.shape[0]-1)

    This is taken from Pylearn2 (https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/expr/normalize.py)
    """
    # doesn't work for even n
    if n % 2 == 0:
        log.error("Cross channel normalization only works for odd n now. N was %s", str(n))
        raise NotImplementedError("Cross channel normalization only works for odd n now. N was %s" % str(n))

    half = n // 2
    sq = T.sqr(bc01)
    b, ch, r, c = bc01.shape
    extra_channels = T.alloc(0., b, ch + 2 * half, r, c)
    sq = T.set_subtensor(extra_channels[:, half:half + ch, :, :], sq)
    scale = k
    for i in xrange(n):
        scale += alpha * sq[:, i:i + ch, :, :]
    scale = scale ** beta

    return bc01 / scale

def cross_channel_normalization_c01b(c01b, alpha=1e-4, k=2, beta=0.75, n=5):
    """
    C01B format (channels, rows, cols, batch) version of Cross Channel Normalization.

    See "ImageNet Classification with Deep Convolutional Neural Networks"
    Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton
    NIPS 2012
    Section 3.3, Local Response Normalization

    f(c01b)_[i,j,k,l] = c01b[i,j,k,l] / scale[i,j,k,l]
    scale[i,j,k,l] = (k + sqr(c01b)[clip(i-n/2):clip(i+n/2),j,k,l].sum())^beta
    clip(i) = T.clip(i, 0, c01b.shape[0]-1)

    This is taken from Pylearn2 (https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/expr/normalize.py)
    """
    # doesn't work for even n
    if n % 2 == 0:
        log.error("Cross channel normalization only works for odd n now. N was %s", str(n))
        raise NotImplementedError("Cross channel normalization only works for odd n now. N was %s" % str(n))

    half = n // 2
    sq = T.sqr(c01b)
    ch, r, c, b = c01b.shape
    extra_channels = T.alloc(0., ch + 2 * half, r, c, b)
    sq = T.set_subtensor(extra_channels[half:half + ch, :, :, :], sq)
    scale = k
    for i in xrange(n):
        scale += alpha * sq[i:i + ch, :, :, :]
    scale = scale ** beta

    return c01b / scale