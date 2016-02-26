"""
Provides various methods for neural net layers.

Based on code from theano_alexnet
(https://github.com/uoguelph-mlrg/theano_alexnet)
"""
# standard libraries
import logging
# third party libraries
from theano.tensor import (concatenate, cast, sqr, alloc, set_subtensor)

log = logging.getLogger(__name__)

def mirror_images(input, image_shape, cropsize, rand, flag_rand):
    """
    This takes an input batch of images (normally the input to a convolutional net),
    and augments them by mirroring and concatenating.

    Parameters
    ----------
    input : Tensor4D
        The input 4D tensor of images.
    image_shape : tuple
        The shape of the 4D tensor input.
    cropsize : int
        What size to crop to.
    rand : vector
        A vector representing a random array for cropping/mirroring the data.
    flag_rand : bool
        Whether to randomize the mirror.

    Returns
    -------
    Tensor4D
        Tensor4D representing the mirrored/concatenated input.
    """
    # The random mirroring and cropping in this function is done for the
    # whole batch.

    # trick for random mirroring
    mirror = input[:, :, ::-1, :]
    input = concatenate([input, mirror], axis=0)

    # crop images
    center_margin = (image_shape[2] - cropsize) / 2

    if flag_rand:
        mirror_rand = cast(rand[2], 'int32')
        crop_xs = cast(rand[0] * center_margin * 2, 'int32')
        crop_ys = cast(rand[1] * center_margin * 2, 'int32')
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
    This helper method uses dimshuffle on a 4D input tensor assumed to be bc01 format (batch, channels, rows, cols)
    and outputs it in c01b format (channels, rows, cols, batch). This operation is used for convolutions.

    Parameters
    ----------
    input : Tensor4D
        A 4D input tensor assumed to be in bc01 ordering. (batch, channels, rows, cols)

    Returns
    -------
    Tensor4D
        A 4D tensor in c01b ordering. (channels, rows, cols, batch)
    """
    # make sure it is a 4d tensor:
    assert input.ndim == 4
    # return the shuffle
    return input.dimshuffle(1, 2, 3, 0)

def c01b_to_bc01(input):
    """
    This helper method uses dimshuffle on a 4D input tensor assumed to be c01b format (channels, rows, cols, batch)
    and outputs it in bc01 format (batch, channels, rows, cols). This operation is used for convolutions.

    Parameters
    ----------
    input : Tensor4D
        A 4D input tensor assumed to be in c01b ordering. (channels, rows, cols, batch)

    Returns
    -------
    Tensor4D
        A 4D tensor in bc01 ordering. (batch, channels, rows, cols)
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
    sq = sqr(bc01)
    b, ch, r, c = bc01.shape
    extra_channels = alloc(0., b, ch + 2 * half, r, c)
    sq = set_subtensor(extra_channels[:, half:half + ch, :, :], sq)
    scale = k
    for i in iter(range(n)):
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
    sq = sqr(c01b)
    ch, r, c, b = c01b.shape
    extra_channels = alloc(0., ch + 2 * half, r, c, b)
    sq = set_subtensor(extra_channels[half:half + ch, :, :, :], sq)
    scale = k
    for i in iter(range(n)):
        scale += alpha * sq[i:i + ch, :, :, :]
    scale = scale ** beta

    return c01b / scale
