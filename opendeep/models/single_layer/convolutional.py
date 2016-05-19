"""
This module provides the base convolutional layers.

.. note::

    To use CuDNN wrapping, you must install the appropriate .h and .so files for theano as described here:
    http://deeplearning.net/software/theano/library/sandbox/cuda/dnn.html

"""
from __future__ import division
# standard libraries
import logging
from collections import Iterable
# third party libraries
from theano import config
from theano.compat.python2x import OrderedDict
from theano.tensor.nnet import conv2d
from theano.tensor.nnet.abstract_conv import get_conv_output_shape, get_conv_shape_1axis
import theano.sandbox.rng_mrg as RNG_MRG
# internal references
from opendeep.models.model import Model
from opendeep.utils.activation import get_activation_function
from opendeep.utils.conv1d_implementations import get_conv1d_function
from opendeep.utils.decorators import inherit_docs
from opendeep.utils.weights import (get_weights, get_bias)

log = logging.getLogger(__name__)

# flag for having NVIDIA's CuDNN library.
has_cudnn = True
try:
    from theano.sandbox.cuda import dnn
    has_cudnn = dnn.dnn_available()
except ImportError as e:
    has_cudnn = False
    log.warning("Could not import CuDNN from theano. For fast convolutions, "
                "please install it like so: http://deeplearning.net/software/theano/library/sandbox/cuda/dnn.html")

# Some convolution operations only work on the GPU, so do a check here:
if not config.device.startswith('gpu'):
    log.warning("You should consider using a GPU, unless this is a small toy algorithm for fun. "
                "Please enable the GPU in Theano via these instructions: "
                "http://deeplearning.net/software/theano/tutorial/using_gpu.html")

# To use the fastest convolutions possible, need to set the Theano flag as described here:
# http://benanne.github.io/2014/12/09/theano-metaopt.html
# make it THEANO_FLAGS=optimizer_including=conv_meta
# OR you could set the .theanorc file with [global]optimizer_including=conv_meta
if config.optimizer_including != "conv_meta":
    log.warning("Theano flag optimizer_including is not conv_meta (found %s)! "
                "To have Theano cherry-pick the best convolution implementation, please set "
                "optimizer_including=conv_meta either in THEANO_FLAGS or in the .theanorc file!"
                % str(config.optimizer_including))

PADDING_STRINGS = ['valid', 'full', 'half']

def get_conv_shape(input_shape, filter_shape, padding, stride):
    """
    Helper method to calculate the shapes post-convolution operation given input parameters. This isn't used
    for our output_size calculations because Theano provides a function specific to its conv op.
    """
    if isinstance(input_shape, Iterable):
        shape = get_conv_output_shape(input_shape, filter_shape, padding, stride)
    else:
        shape = get_conv_shape_1axis(input_shape, filter_shape, padding, stride)
    return shape
    # batch_size = input_shape[0]
    # num_filters = filter_shape[0]
    # return ((batch_size, num_filters) +
    #         tuple(conv_output_length(input, filter, stride, p)
    #               for input, filter, stride, p
    #               in zip(input_shape[2:], self.filter_size,
    #                      self.stride, pad)))

@inherit_docs
class Conv1D(Model):
    """
    A 1-dimensional convolutional layer (taken from Sander Dieleman's Lasagne framework)
    (https://github.com/benanne/Lasagne/blob/master/lasagne/theano_extensions/conv.py)

    This means the input is a 3-dimensional tensor of form (batch, channel, input)
    """
    def __init__(self, inputs=None, params=None, outdir='outputs/conv1d',
                 n_filters=None, filter_size=None, stride=None, padding='valid',
                 weights_init='uniform', weights_interval='glorot', weights_mean=0, weights_std=5e-3,
                 bias_init=0,
                 activation='elu',
                 convolution='mc0',
                 mrg=RNG_MRG.MRG_RandomStreams(1),
                 **kwargs):
        """
        Initialize a 1-D convolutional layer.

        Parameters
        ----------
        inputs : tuple(shape, `Theano.TensorType`)
            The dimensionality of the inputs for this model, and the routing information for the model
            to accept inputs from elsewhere. `shape` will be a monad tuple representing known
            sizes for each dimension in the `Theano.TensorType`. Shape of the incoming data:
            (batch_size, num_channels, data_dimensionality). Most likely, your channels
            will be 1. For example, batches of text will be of the form (N, 1, D) where N=examples in minibatch and
            D=dimensionality (chars, words, etc.)
        params : Dict(string_name: theano SharedVariable), optional
            A dictionary of model parameters (shared theano variables) that you should use when constructing
            this model (instead of initializing your own shared variables). This parameter is useful when you want to
            have two versions of the model that use the same parameters - such as siamese networks or pretraining some
            weights.
        outdir : str or None
            The directory you want outputs (parameters, images, etc.) to save to. If None, nothing will
            be saved.
        n_filters : int
            The number of filters to use (convolution kernels).
        filter_size : int
            The size of the convolution filter.
        stride : int
            The distance between the receptive field centers of neighboring units. This is the 'stride' of the
            convolution operation.
        padding : str, one of 'valid', 'full', 'same', or int
            A string indicating the convolution border mode.
            If 'valid', the convolution is only computed where the input and the
            filter fully overlap.
            If 'full', the convolution is computed wherever the input and the
            filter overlap by at least one position.
            An int specifies the amount of padding to add manually.
        weights_init : str
            Determines the method for initializing model weights. See opendeep.utils.nnet for options.
        weights_interval : str or float
            If Uniform `weights_init`, the +- interval to use. See opendeep.utils.nnet for options.
        weights_mean : float
            If Gaussian `weights_init`, the mean value to use.
        weights_std : float
            If Gaussian `weights_init`, the standard deviation to use.
        bias_init : float
            The initial value to use for the bias parameter. Most often, the default of 0.0 is preferred.
        activation : str or Callable
            The activation function to apply to the layer. See opendeep.utils.activation for options.
        convolution : str or Callable
            The 1-dimensional convolution implementation to use. The default of 'mc0' is normally fine. See
            opendeep.utils.conv1d_implementations for alternatives. (This is necessary because Theano only
            supports 2D convolutions at the moment).
        mrg : random
            A random number generator that is used when adding noise.
            I recommend using Theano's sandbox.rng_mrg.MRG_RandomStreams.

        Notes
        -----
        Theano's default convolution function (`theano.tensor.nnet.conv.conv2d`)
        does not support the 'same' border mode by default. This layer emulates
        it by performing a 'full' convolution and then cropping the result, which
        may negatively affect performance.
        """
        initial_parameters = locals().copy()
        initial_parameters.pop('self')
        super(Conv1D, self).__init__(**initial_parameters)
        if self.inputs is None:
            return

        ##################
        # specifications #
        ##################
        # grab info from the inputs_hook, or from parameters
        # expect input to be in the form (B, C, I) (batch, channel, input data)
        # inputs_hook is a tuple of (Shape, Input)
        # self.inputs is a list of all the input expressions (we enforce only 1, so self.inputs[0] is the input)
        input_shape, self.input = self.inputs[0]
        assert self.input.ndim == 3, "Expected 3D input variable with form (batch, channel, input_data)"
        assert len(input_shape) == 3, "Expected 3D input shape with form (batch, channel, input_data)"

        n_channels = input_shape[1]

        filter_shape = (n_filters, n_channels, filter_size)

        # activation function!
        activation_func = get_activation_function(activation)

        # convolution function!
        convolution_func = get_conv1d_function(convolution)

        outshape = get_conv_shape(input_shape=input_shape[2], filter_shape=filter_size, padding=padding, stride=stride)

        self.output_size = (input_shape[0], n_filters, outshape)

        ##########
        # Params #
        ##########
        W = self.params.get(
            "W",
            get_weights(weights_init=weights_init,
                        shape=filter_shape,
                        name="W",
                        rng=mrg,
                        # if gaussian
                        mean=weights_mean,
                        std=weights_std,
                        # if uniform
                        interval=weights_interval)
        )

        b = self.params.get(
            "b",
            get_bias(shape=(n_filters,), name="b", init_values=bias_init)
        )

        # Finally have the two parameters!
        self.params = OrderedDict([("W", W), ("b", b)])

        ########################
        # Computational Graph! #
        ########################
        if padding in PADDING_STRINGS or isinstance(padding, int):
            conved = convolution_func(self.input,
                                      W,
                                      subsample=(stride,),
                                      input_shape=input_shape,
                                      filter_shape=filter_shape,
                                      border_mode=padding)
        else:
            log.error("Invalid padding: '{!s}'. Expected int or one of {!s}".format(padding, PADDING_STRINGS))
            raise RuntimeError("Invalid padding: '{!s}'. Expected int or one of {!s}".format(padding, PADDING_STRINGS))

        self.output = activation_func(conved + b.dimshuffle('x', 0, 'x'))

    def get_inputs(self):
        return [self.input]

    def get_outputs(self):
        return self.output

    def get_params(self):
        return self.params


@inherit_docs
class Conv2D(Model):
    """
    A 2-dimensional convolutional layer (taken from Sander Dieleman's Lasagne framework)
    (https://github.com/benanne/Lasagne/blob/master/lasagne/theano_extensions/conv.py)
    """
    def __init__(self, inputs=None, params=None, outdir='outputs/conv2d',
                 n_filters=None, filter_size=None, padding='valid', stride=(1, 1),
                 weights_init='uniform', weights_interval='glorot', weights_mean=0, weights_std=5e-3,
                 bias_init=0,
                 activation='elu',
                 convolution='conv2d',
                 mrg=RNG_MRG.MRG_RandomStreams(1),
                 **kwargs):
        """
        Initialize a 2-dimensional convolutional layer.

        Parameters
        ----------
        inputs : tuple(shape, `Theano.TensorType`)
            The dimensionality of the inputs for this model, and the routing information for the model
            to accept inputs from elsewhere. `shape` will be a monad tuple representing known
            sizes for each dimension in the `Theano.TensorType`. Shape of the incoming data:
            (batch_size, num_channels, input_height, input_width).
            If input_size is None, it can be inferred. However, padding can't be 'same'.
        params : Dict(string_name: theano SharedVariable), optional
            A dictionary of model parameters (shared theano variables) that you should use when constructing
            this model (instead of initializing your own shared variables). This parameter is useful when you want to
            have two versions of the model that use the same parameters - such as siamese networks or pretraining some
            weights.
        outdir : str or None
            The directory you want outputs (parameters, images, etc.) to save to. If None, nothing will
            be saved.
        n_filters : int
            The number of filters to use (convolution kernels).
        filter_size : tuple(int) or int
            (filter_height, filter_width). If it is an int, size will be duplicated across height and width.
        padding : str, one of 'valid', 'full', 'half', or int or tuple(int)
            A string indicating the convolution border mode.
            If 'valid', the convolution is only computed where the input and the
            filter fully overlap.
            If 'full', the convolution is computed wherever the input and the
            filter overlap by at least one position.
            If int, symmetric padding defined by the integer.
            If tuple(int), height, width padding defined by each element.
        stride : tuple(int)
            The distance between the receptive field centers of neighboring units. This is the 'stride' of the
            convolution operation.
        weights_init : str
            Determines the method for initializing model weights. See opendeep.utils.nnet for options.
        weights_interval : str or float
            If Uniform `weights_init`, the +- interval to use. See opendeep.utils.nnet for options.
        weights_mean : float
            If Gaussian `weights_init`, the mean value to use.
        weights_std : float
            If Gaussian `weights_init`, the standard deviation to use.
        bias_init : float
            The initial value to use for the bias parameter. Most often, the default of 0.0 is preferred.
        activation : str or Callable
            The activation function to apply to the layer. See opendeep.utils.activation for options.
        convolution : str or Callable
            The 2-dimensional convolution implementation to use. The default of 'conv2d' is normally fine because it
            uses theano's tensor.nnet.conv.conv2d, which cherry-picks the best implementation with a meta-optimizer if
            you set the theano configuration flag 'optimizer_including=conv_meta'. Otherwise, you could pass a
            callable function, such as cudnn or cuda-convnet if you don't want to use the meta-optimizer, or write your
            own. Your own function should expect parameters in this order:
        mrg : random
            A random number generator that is used when adding noise.
            I recommend using Theano's sandbox.rng_mrg.MRG_RandomStreams.
        """
        super(Conv2D, self).__init__(**{arg: val for (arg, val) in locals().items() if arg is not 'self'})

        ##################
        # specifications #
        ##################
        # expect input to be in the form (B, C, 0, 1) (batch, channel, rows, cols)
        # self.inputs is a list of all the input expressions (we enforce only 1, so self.inputs[0] is the input)
        input_shape, self.input = self.inputs[0]
        assert self.input.ndim == 4, "Expected 4D input variable with form (batch, channel, rows, cols)"
        assert len(input_shape) == 4, "Expected 4D input shape with form (batch, channel, rows, cols)"

        n_channels = input_shape[1]

        if isinstance(filter_size, int):
            filter_size = (filter_size, )*2
        if isinstance(padding, int):
            padding = (padding, )*2
        if isinstance(stride, int):
            stride = (stride, )*2

        # activation function!
        activation_func = get_activation_function(activation)

        # convolution function!
        if convolution == 'conv2d':
            # using the theano flag optimizer_including=conv_meta will let this conv function optimize itself.
            convolution_func = conv2d
        else:
            assert callable(convolution), "Input convolution was not 'conv2d' and was not Callable."
            convolution_func = convolution

        # filter shape should be in the form (num_filters, num_channels, filter_size[0], filter_size[1])
        filter_shape = (n_filters, n_channels) + filter_size

        self.output_size = get_conv_shape(
            input_shape=input_shape, filter_shape=filter_shape, padding=padding, stride=stride
        )

        ##########
        # Params #
        ##########
        W = self.params.get(
            "W",
            get_weights(weights_init=weights_init,
                        shape=filter_shape,
                        name="W",
                        rng=mrg,
                        # if gaussian
                        mean=weights_mean,
                        std=weights_std,
                        # if uniform
                        interval=weights_interval)
        )

        b = self.params.get(
            "b",
            get_bias(shape=(n_filters, ), name="b", init_values=bias_init)
        )

        # Finally have the two parameters!
        self.params = OrderedDict([("W", W), ("b", b)])

        ########################
        # Computational Graph! #
        ########################
        if padding in PADDING_STRINGS or isinstance(padding, int) or isinstance(padding, Iterable):
            conved = convolution_func(self.input,
                                      W,
                                      subsample=stride,
                                      input_shape=input_shape,
                                      filter_shape=filter_shape,
                                      border_mode=padding)
        else:
            log.error("Invalid padding: '{!s}'. Expected int or one of {!s}".format(padding, PADDING_STRINGS))
            raise RuntimeError("Invalid padding: '{!s}'. Expected int, tuple(int), or one of {!s}".format(
                padding, PADDING_STRINGS))

        self.output = activation_func(conved + b.dimshuffle('x', 0, 'x', 'x'))

    def get_inputs(self):
        return [self.input]

    def get_outputs(self):
        return self.output

    def get_params(self):
        return self.params
