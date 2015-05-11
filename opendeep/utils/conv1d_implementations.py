"""
Alternative 1-dimensional convolution implementations.

These 1-dimensional convolutions were taken from the Sander Dieleman's Lasagne framework:
https://github.com/benanne/Lasagne/blob/master/lasagne/theano_extensions/conv.py
"""
__authors__ = "Sander Dieleman"
__credits__ = ["Sander Dieleman", "Daniel Nouri", "Colin Raffel", "Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
# third party libraries
import numpy
import theano
from theano.compat import six
import theano.tensor as T

log = logging.getLogger(__name__)

# To use the fastest convolutions possible, need to set the Theano flag as described here:
# http://benanne.github.io/2014/12/09/theano-metaopt.html
# make it THEANO_FLAGS=optimizer_including=conv_meta
# OR you could set the .theanorc file with [global]optimizer_including=conv_meta
if theano.config.optimizer_including != "conv_meta":
    log.warning("Theano flag optimizer_including is not conv_meta (found %s)! "
                "To have Theano cherry-pick the best convolution implementation, please set "
                "optimizer_including=conv_meta either in THEANO_FLAGS or in the .theanorc file!"
                % str(theano.config.optimizer_including))

# 1D convolutions
# These convolutions assume the input is shaped like (B, C, I), which is (Batch, Channel, Input data).
# Most likely, your channel will be 1.
# For example, batches of text will be of the form (N, 1, D) where N=examples in minibatch and
# D=dimensionality (chars, words, etc.)

def conv1d_sc(input, filters, image_shape=None, filter_shape=None,
              border_mode='valid', subsample=(1,)):
    """
    Using conv2d with a single input channel.

    border_mode has to be 'valid' at the moment.
    """
    if border_mode != 'valid':
        log.error("Unsupported border_mode for conv1d_sc: "
                  "%s" % border_mode)
        raise RuntimeError("Unsupported border_mode for conv1d_sc: "
                           "%s" % border_mode)

    if image_shape is None:
        image_shape_sc = None
    else:
        # (b, c, i0) to (b, 1, c, i0)
        image_shape_sc = (image_shape[0], 1, image_shape[1], image_shape[2])

    if filter_shape is None:
        filter_shape_sc = None
    else:
        filter_shape_sc = (filter_shape[0], 1, filter_shape[1],
                           filter_shape[2])

    input_sc = input.dimshuffle(0, 'x', 1, 2)
    # We need to flip the channels dimension because it will be convolved over.
    filters_sc = filters.dimshuffle(0, 'x', 1, 2)[:, :, ::-1, :]

    conved = T.nnet.conv2d(input_sc, filters_sc, image_shape=image_shape_sc,
                           filter_shape=filter_shape_sc,
                           subsample=(1, subsample[0]))
    return conved[:, :, 0, :]  # drop the unused dimension


def conv1d_mc0(input, filters, image_shape=None, filter_shape=None,
               border_mode='valid', subsample=(1,)):
    """
    Using conv2d with width == 1.
    """
    if image_shape is None:
        image_shape_mc0 = None
    else:
        # (b, c, i0) to (b, c, 1, i0)
        image_shape_mc0 = (image_shape[0], image_shape[1], 1, image_shape[2])

    if filter_shape is None:
        filter_shape_mc0 = None
    else:
        filter_shape_mc0 = (filter_shape[0], filter_shape[1], 1,
                            filter_shape[2])

    input_mc0 = input.dimshuffle(0, 1, 'x', 2)
    filters_mc0 = filters.dimshuffle(0, 1, 'x', 2)

    conved = T.nnet.conv2d(
        input_mc0, filters_mc0, image_shape=image_shape_mc0,
        filter_shape=filter_shape_mc0, subsample=(1, subsample[0]),
        border_mode=border_mode)
    return conved[:, :, 0, :]  # drop the unused dimension


def conv1d_mc1(input, filters, image_shape=None, filter_shape=None,
               border_mode='valid', subsample=(1,)):
    """
    Using conv2d with height == 1.
    """
    if image_shape is None:
        image_shape_mc1 = None
    else:
        # (b, c, i0) to (b, c, i0, 1)
        image_shape_mc1 = (image_shape[0], image_shape[1], image_shape[2], 1)

    if filter_shape is None:
        filter_shape_mc1 = None
    else:
        filter_shape_mc1 = (filter_shape[0], filter_shape[1],
                            filter_shape[2], 1)

    input_mc1 = input.dimshuffle(0, 1, 2, 'x')
    filters_mc1 = filters.dimshuffle(0, 1, 2, 'x')

    conved = T.nnet.conv2d(
        input_mc1, filters_mc1, image_shape=image_shape_mc1,
        filter_shape=filter_shape_mc1, subsample=(subsample[0], 1),
        border_mode=border_mode)
    return conved[:, :, :, 0]  # drop the unused dimension


def conv1d_unstrided(input, filters, image_shape, filter_shape,
                     border_mode='valid', subsample=(1,),
                     implementation=conv1d_sc):
    """
    Perform a strided 1D convolution by reshaping input and filters so that the
    stride becomes 1. This function requires that the filter length is a
    multiple of the stride. It also truncates the input to have a length
    that is a multiple of the stride.

    border_mode has to be 'valid' at the moment.
    """
    batch_size, num_input_channels, input_length = image_shape
    num_filters, num_input_channels_, filter_length = filter_shape
    stride = subsample[0]

    if filter_length % stride > 0:
        log.error("Filter length (%d) is not a multiple of the "
                  "stride (%d)" % (filter_length, stride))
        raise RuntimeError("Filter length (%d) is not a multiple of the "
                           "stride (%d)" % (filter_length, stride))
    # TODO: test if this works for border_mode='full'
    assert border_mode == 'valid'

    num_steps = filter_length // stride

    # input sizes need to be multiples of the strides,
    # truncate to correct sizes.
    truncated_length = (input_length // stride) * stride
    input_truncated = input[:, :, :truncated_length]

    r_input_shape = (batch_size, num_input_channels,
                     truncated_length // stride, stride)
    r_input = input_truncated.reshape(r_input_shape)

    # fold strides into the feature maps dimension (input)
    r_input_folded_shape = (batch_size, num_input_channels * stride,
                            truncated_length // stride)
    r_input_folded = r_input.dimshuffle(
        0, 1, 3, 2).reshape(r_input_folded_shape)

    r_filter_shape = (num_filters, num_input_channels, num_steps, stride)
    r_filters_flipped = filters[:, :, ::-1].reshape(r_filter_shape)

    # fold strides into the feature maps dimension (filters)
    r_filter_folded_shape = (num_filters, num_input_channels * stride,
                             num_steps)
    r_filters_flipped_folded = r_filters_flipped.dimshuffle(
        0, 1, 3, 2).reshape(r_filter_folded_shape)
    r_filters_folded = r_filters_flipped_folded[:, :, ::-1]  # unflip

    return implementation(r_input_folded, r_filters_folded,
                          r_input_folded_shape, r_filter_folded_shape,
                          border_mode, subsample=(1,))


def conv1d_sd(input, filters, image_shape, filter_shape, border_mode='valid',
              subsample=(1,)):
    """
    Using a single dot product.

    border_mode has to be 'valid' at the moment.
    """
    if border_mode != 'valid':
        log.error("Unsupported border_mode for conv1d_sd: "
                  "%s" % border_mode)
        raise RuntimeError("Unsupported border_mode for conv1d_sd: "
                           "%s" % border_mode)

    batch_size, num_input_channels, input_length = image_shape
    num_filters, num_input_channels_, filter_length = filter_shape
    stride = subsample[0]

    if filter_length % stride > 0:
        raise RuntimeError("Filter length (%d) is not a multiple of the "
                           "stride (%d)" % (filter_length, stride))

    num_steps = filter_length // stride
    output_length = (input_length - filter_length + stride) // stride

    # pad the input so all the shifted dot products fit inside.
    # shape is (b, c, l)
    padded_length = ((input_length // filter_length) * filter_length +
                     (num_steps - 1) * stride)

    # at this point, it is possible that the padded_length is SMALLER than the
    # input size. so then we have to truncate first.
    truncated_length = min(input_length, padded_length)
    input_truncated = input[:, :, :truncated_length]

    input_padded_shape = (batch_size, num_input_channels, padded_length)
    input_padded = T.zeros(input_padded_shape)
    input_padded = T.set_subtensor(input_padded[:, :, :truncated_length],
                                   input_truncated)

    inputs = []
    for num in range(num_steps):
        shift = num * stride
        length = (padded_length - shift) // filter_length

        r_input_shape = (batch_size, num_input_channels, length, filter_length)
        r_input = input_padded[
            :, :, shift:length * filter_length + shift].reshape(r_input_shape)

        inputs.append(r_input)

    inputs_stacked = T.stack(*inputs)  # shape is (n, b, c, w, f)
    filters_flipped = filters[:, :, ::-1]

    r_conved = T.tensordot(inputs_stacked, filters_flipped,
                           numpy.asarray([[2, 4], [1, 2]], dtype=theano.config.floatX))
    # resulting shape is (n, b, w, n_filters)
    # output needs to be (b, n_filters, w * n)
    r_conved = r_conved.dimshuffle(1, 3, 2, 0)  # (b, n_filters, w, n)
    conved = r_conved.reshape((r_conved.shape[0], r_conved.shape[1],
                               r_conved.shape[2] * r_conved.shape[3]))
    # result is (b, n_f, l)

    # remove padding
    return conved[:, :, :output_length]


def conv1d_md(input, filters, image_shape, filter_shape, border_mode='valid',
              subsample=(1,)):
    """
    Using multiple dot products.

    border_mode has to be 'valid' at the moment.
    """
    if border_mode != 'valid':
        log.error("Unsupported border_mode for conv1d_md: "
                  "%s" % border_mode)
        raise RuntimeError("Unsupported border_mode for conv1d_md: "
                           "%s" % border_mode)

    batch_size, num_input_channels, input_length = image_shape
    num_filters, num_input_channels_, filter_length = filter_shape
    stride = subsample[0]

    if filter_length % stride > 0:
        log.error("Filter length (%d) is not a multiple of the "
                  "stride (%d)" % (filter_length, stride))
        raise RuntimeError("Filter length (%d) is not a multiple of the "
                           "stride (%d)" % (filter_length, stride))

    num_steps = filter_length // stride
    output_length = (input_length - filter_length + stride) // stride
    output_shape = (batch_size, num_filters, output_length)

    filters_flipped = filters[:, :, ::-1]

    conved = T.zeros(output_shape)

    for num in range(num_steps):
        shift = num * stride
        length = (input_length - shift) // filter_length

        if length == 0:
            # we can safely skip this product, it doesn't contribute to the
            # final convolution.
            continue

        r_input_shape = (batch_size, num_input_channels, length, filter_length)
        r_input = input[
            :, :, shift:length * filter_length + shift].reshape(r_input_shape)

        # shape (b, l, n_filters)
        r_conved = T.tensordot(r_input, filters_flipped,
                               numpy.asarray([[1, 3], [1, 2]], dtype=theano.config.floatX))
        r_conved = r_conved.dimshuffle(0, 2, 1)  # shape is (b, n_filters, l)
        conved = T.set_subtensor(conved[:, :, num::num_steps], r_conved)

    return conved


# TODO: conv1d_md_channelslast? (from lasagne)

############# keep conv1d functions above this line, and add them to the dictionary below #############
# this is a dictionary containing a string keyword mapping to the conv1d function -
# used for get_conv1d_function(name)
_conv1d = {'sc': conv1d_sc,
           'mc0': conv1d_mc0,
           'mc1': conv1d_mc1,
           'md': conv1d_md,
           'sd': conv1d_sd,
           'unstrided': conv1d_unstrided}

def get_conv1d_function(name):
    """
    This helper method returns the appropriate 1-dimensional convolution function given a string name.
    It looks up the appropriate function from the internal _conv1d dictionary.

    Parameters
    ----------
    name : str or Callable
        String representation of the function you want. If callable, assumes you are using your own function and
        returns that.

    Returns
    -------
    function
        The appropriate 1-dimensional convolution function, or raise NotImplementedError if it isn't found.

    Raises
    ------
    NotImplementedError
        When the name cannot be found in the internal dictionary.
    """
    # return the function itself if it is a Callable
    if callable(name):
        return name
    # otherwise if it is a string
    elif isinstance(name, six.string_types):
        # standardize the input to be lowercase
        name = name.lower()
        # grab the appropriate activation function from the dictionary of activations
        func = _conv1d.get(name)
        # if it couldn't find the function (key didn't exist), raise a NotImplementedError
        if func is None:
            log.critical("Did not recognize conv1d %s! Please use one of: ", str(name), str(_conv1d.keys()))
            raise NotImplementedError(
                "Did not recognize conv1d {0!s}! Please use one of: {1!s}".format(name, _conv1d.keys())
            )
        # return the found function
        return func
    # else we don't know what to do so throw error
    else:
        log.critical("Convolution function not implemented for %s with type %s", str(name), str(type(name)))
        raise NotImplementedError("Convolution function not implemented for %s with type %s",
                                  str(name), str(type(name)))