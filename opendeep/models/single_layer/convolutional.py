"""
This module provides the base convolutional layers.

.. note::

    To use CuDNN wrapping, you must install the appropriate .h and .so files for theano as described here:
    http://deeplearning.net/software/theano/library/sandbox/cuda/dnn.html

"""

__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Lasagne", "Weiguang Ding", "Ruoyan Wang", "Fei Mao", "Graham Taylor", "Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
# third party libraries
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
import theano.sandbox.rng_mrg as RNG_MRG
# internal references
from opendeep.models.model import Model
from opendeep.utils.activation import get_activation_function
from opendeep.utils.conv1d_implementations import get_conv1d_function
from opendeep.utils.decorators import inherit_docs
from opendeep.utils.nnet import get_weights, get_weights_gaussian, get_bias, cross_channel_normalization_bc01

log = logging.getLogger(__name__)

# flag for having NVIDIA's CuDNN library.
has_cudnn = True
try:
    from theano.sandbox.cuda import dnn
except ImportError as e:
    has_cudnn = False
    log.warning("Could not import CuDNN from theano. For fast convolutions, "
                "please install it like so: http://deeplearning.net/software/theano/library/sandbox/cuda/dnn.html")

# Some convolution operations only work on the GPU, so do a check here:
if not theano.config.device.startswith('gpu'):
    log.warning("You should reeeeeaaaally consider using a GPU, unless this is a small toy algorithm for fun. "
                "Please enable the GPU in Theano via these instructions: "
                "http://deeplearning.net/software/theano/tutorial/using_gpu.html")

# To use the fastest convolutions possible, need to set the Theano flag as described here:
# http://benanne.github.io/2014/12/09/theano-metaopt.html
# make it THEANO_FLAGS=optimizer_including=conv_meta
# OR you could set the .theanorc file with [global]optimizer_including=conv_meta
if theano.config.optimizer_including != "conv_meta":
    log.warning("Theano flag optimizer_including is not conv_meta (found %s)! "
                "To have Theano cherry-pick the best convolution implementation, please set "
                "optimizer_including=conv_meta either in THEANO_FLAGS or in the .theanorc file!"
                % str(theano.config.optimizer_including))


@inherit_docs
class Conv1D(Model):
    """
    A 1-dimensional convolutional layer (taken from Sander Dieleman's Lasagne framework)
    (https://github.com/benanne/Lasagne/blob/master/lasagne/theano_extensions/conv.py)

    This means the input is a 3-dimensional tensor of form (batch, channel, input)
    """
    def __init__(self, inputs_hook=None, params_hook=None, outdir='outputs/conv1d',
                 input_size=None, filter_shape=None, stride=None, border_mode='valid',
                 weights_init='uniform', weights_interval='montreal', weights_mean=0, weights_std=5e-3,
                 bias_init=0,
                 activation='rectifier',
                 convolution='mc0',
                 mrg=RNG_MRG.MRG_RandomStreams(1)):
        """
        Initialize a 1-D convolutional layer.

        Parameters
        ----------
        inputs_hook : Tuple of (shape, variable)
            Routing information for the model to accept inputs from elsewhere. This is used for linking
            different models together. For now, it needs to include the shape information.
        params_hook : List(theano shared variable)
            A list of model parameters (shared theano variables) that you should use when constructing
            this model (instead of initializing your own shared variables).
        outdir : str
            The directory you want outputs (parameters, images, etc.) to save to. If None, nothing will
            be saved.
        input_size : tuple
            Shape of the incoming data: (batch_size, num_channels, data_dimensionality). Most likely, your channels
            will be 1. For example, batches of text will be of the form (N, 1, D) where N=examples in minibatch and
            D=dimensionality (chars, words, etc.)
        filter_shape : tuple
            (num_filters, num_channels, filter_length). This is also the shape of the weights matrix.
        stride : int
            The distance between the receptive field centers of neighboring units. This is the 'stride' of the
            convolution operation.
        border_mode : str, one of 'valid', 'full', 'same'
            A string indicating the convolution border mode.
            If 'valid', the convolution is only computed where the input and the
            filter fully overlap.
            If 'full', the convolution is computed wherever the input and the
            filter overlap by at least one position.
            If 'same', the convolution is computed wherever the input and the
            filter overlap by at least half the filter size, when the filter size
            is odd. In practice, the input is zero-padded with half the filter size
            at the beginning and half at the end (or one less than half in the case
            of an even filter size). This results in an output length that is the
            same as the input length (for both odd and even filter sizes).
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
        super(Conv1D, self).__init__(**{arg: val for (arg, val) in locals().iteritems() if arg is not 'self'})

        ##################
        # specifications #
        ##################
        # grab info from the inputs_hook, or from parameters
        # expect input to be in the form (B, C, I) (batch, channel, input data)
        # inputs_hook is a tuple of (Shape, Input)
        if self.inputs_hook is not None:
            # make sure inputs_hook is a tuple
            assert len(self.inputs_hook) == 2, "expecting inputs_hook to be tuple of (shape, input)"
            self.input = inputs_hook[1]
        else:
            # make the input a symbolic matrix
            self.input = T.ftensor3('X')

        # activation function!
        activation_func = get_activation_function(activation)

        # convolution function!
        convolution_func = get_conv1d_function(convolution)

        # filter shape should be in the form (num_filters, num_channels, filter_length)
        num_filters = filter_shape[0]
        filter_length = filter_shape[2]

        ################################################
        # Params - make sure to deal with params_hook! #
        ################################################
        if self.params_hook:
            # make sure the params_hook has W and b
            assert len(self.params_hook) == 2, \
                "Expected 2 params (W and b) for Conv1D, found {0!s}!".format(len(self.params_hook))
            W, b = self.params_hook
        else:
            W = get_weights(weights_init=weights_init,
                            shape=filter_shape,
                            name="W",
                            rng=mrg,
                            # if gaussian
                            mean=weights_mean,
                            std=weights_std,
                            # if uniform
                            interval=weights_interval)

            b = get_bias(shape=(num_filters,), name="b", init_values=bias_init)

        # Finally have the two parameters!
        self.params = [W, b]

        ########################
        # Computational Graph! #
        ########################
        if border_mode in ['valid', 'full']:
            conved = convolution_func(self.input,
                                      W,
                                      subsample=(stride,),
                                      image_shape=self.input_size,
                                      filter_shape=filter_shape,
                                      border_mode=border_mode)
        elif border_mode == 'same':
            conved = convolution_func(self.input,
                                      W,
                                      subsample=(stride,),
                                      image_shape=self.input_size,
                                      filter_shape=filter_shape,
                                      border_mode='full')
            shift = (filter_length - 1) // 2
            conved = conved[:, :, shift:self.input_size[2] + shift]

        else:
            log.error("Invalid border mode: '%s'" % border_mode)
            raise RuntimeError("Invalid border mode: '%s'" % border_mode)

        self.output = activation_func(conved + b.dimshuffle('x', 0, 'x'))

    def get_inputs(self):
        return [self.input]

    def get_outputs(self):
        return self.output

    def get_params(self):
        return self.params

    def save_args(self, args_file="conv1d_config.pkl"):
        super(Conv1D, self).save_args(args_file)


@inherit_docs
class Conv2D(Model):
    """
    A 2-dimensional convolutional layer (taken from Sander Dieleman's Lasagne framework)
    (https://github.com/benanne/Lasagne/blob/master/lasagne/theano_extensions/conv.py)
    """
    def __init__(self, inputs_hook=None, params_hook=None, outdir='outputs/conv2d',
                 input_size=None, filter_shape=None, strides=None, border_mode='valid',
                 weights_init='uniform', weights_interval='montreal', weights_mean=0, weights_std=5e-3,
                 bias_init=0,
                 activation='rectifier',
                 convolution='conv2d',
                 mrg=RNG_MRG.MRG_RandomStreams(1)):
        """
        Initialize a 2-dimensional convolutional layer.

        Parameters
        ----------
        inputs_hook : Tuple of (shape, variable)
            Routing information for the model to accept inputs from elsewhere. This is used for linking
            different models together. For now, it needs to include the shape information.
        params_hook : List(theano shared variable)
            A list of model parameters (shared theano variables) that you should use when constructing
            this model (instead of initializing your own shared variables).
        outdir : str
            The directory you want outputs (parameters, images, etc.) to save to. If None, nothing will
            be saved.
        input_size : tuple
            Shape of the incoming data: (batch_size, num_channels, input_height, input_width).
        filter_shape : tuple
            (num_filters, num_channels, filter_height, filter_width). This is also the shape of the weights matrix.
        stride : int
            The distance between the receptive field centers of neighboring units. This is the 'stride' of the
            convolution operation.
        border_mode : str, one of 'valid', 'full', 'same'
            A string indicating the convolution border mode.
            If 'valid', the convolution is only computed where the input and the
            filter fully overlap.
            If 'full', the convolution is computed wherever the input and the
            filter overlap by at least one position.
            If 'same', the convolution is computed wherever the input and the
            filter overlap by at least half the filter size, when the filter size
            is odd. In practice, the input is zero-padded with half the filter size
            at the beginning and half at the end (or one less than half in the case
            of an even filter size). This results in an output length that is the
            same as the input length (for both odd and even filter sizes).
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
            callable function, such as cudnn or cuda-convnet if you don't want to use the meta-optimizer.
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
        super(Conv2D, self).__init__(**{arg: val for (arg, val) in locals().iteritems() if arg is not 'self'})

        ##################
        # specifications #
        ##################
        # grab info from the inputs_hook, or from parameters
        # expect input to be in the form (B, C, 0, 1) (batch, channel, rows, cols)
        # inputs_hook is a tuple of (Shape, Input)
        if self.inputs_hook:
            # make sure inputs_hook is a tuple
            assert len(self.inputs_hook) == 2, "expecting inputs_hook to be tuple of (shape, input)"
            self.input = inputs_hook[1]
        else:
            # make the input a symbolic matrix
            self.input = T.ftensor4('X')

        # activation function!
        activation_func = get_activation_function(activation)

        # convolution function!
        if convolution == 'conv2d':
            # using the theano flag optimizer_including=conv_meta will let this conv function optimize itself.
            convolution_func = T.nnet.conv2d
        else:
            assert callable(convolution), "Input convolution was not 'conv2d' and was not Callable."
            convolution_func = convolution

        # filter shape should be in the form (num_filters, num_channels, filter_size[0], filter_size[1])
        num_filters = filter_shape[0]
        filter_size = filter_shape[2:3]

        ################################################
        # Params - make sure to deal with params_hook! #
        ################################################
        if self.params_hook:
            # make sure the params_hook has W and b
            assert len(self.params_hook) == 2, \
                "Expected 2 params (W and b) for Conv2D, found {0!s}!".format(len(self.params_hook))
            W, b = self.params_hook
        else:
            W = get_weights(weights_init=weights_init,
                            shape=filter_shape,
                            name="W",
                            rng=mrg,
                            # if gaussian
                            mean=weights_mean,
                            std=weights_std,
                            # if uniform
                            interval=weights_interval)

            b = get_bias(shape=(num_filters, ), name="b", init_values=bias_init)

        # Finally have the two parameters!
        self.params = [W, b]

        ########################
        # Computational Graph! #
        ########################
        if border_mode in ['valid', 'full']:
            conved = convolution_func(self.input,
                                      W,
                                      subsample=strides,
                                      image_shape=self.input_size,
                                      filter_shape=filter_shape,
                                      border_mode=border_mode)
        elif border_mode == 'same':
            conved = convolution_func(self.input,
                                      W,
                                      subsample=strides,
                                      image_shape=self.input_size,
                                      filter_shape=filter_shape,
                                      border_mode='full')
            shift_x = (filter_size[0] - 1) // 2
            shift_y = (filter_size[1] - 1) // 2
            conved = conved[:, :, shift_x:self.input_size[2] + shift_x,
                            shift_y:self.input_size[3] + shift_y]
        else:
            raise RuntimeError("Invalid border mode: '%s'" % border_mode)

        self.output = activation_func(conved + b.dimshuffle('x', 0, 'x', 'x'))

    def get_inputs(self):
        return [self.input]

    def get_outputs(self):
        return self.output

    def get_params(self):
        return self.params

    def save_args(self, args_file="conv2d_config.pkl"):
        super(Conv2D, self).save_args(args_file)


class Conv3D(Model):
    """
    A 3-dimensional convolution layer

    .. todo:: Implement me!
    """
    def __init__(self):
        log.error("Conv3D not implemented yet.")
        raise NotImplementedError("Conv3D not implemented yet.")


@inherit_docs
class ConvPoolLayer(Model):
    """
    This is the ConvPoolLayer used for an AlexNet implementation. It combines a 2D convolution with pooling.
    """
    def __init__(self, inputs_hook=None, params_hook=None, outdir='outputs/convpool',
                 input_size=None, filter_shape=None, convstride=4, padsize=0, group=1,
                 poolsize=3, poolstride=2,
                 weights_init='gaussian', weights_interval='montreal', weights_mean=0, weights_std=.01,
                 bias_init=0,
                 local_response_normalization=False,
                 convolution='conv2d',
                 activation='rectifier',
                 mrg=RNG_MRG.MRG_RandomStreams(1)):
        """
        Initialize a convpool layer.

        Parameters
        ----------
        inputs_hook : Tuple of (shape, variable)
            Routing information for the model to accept inputs from elsewhere. This is used for linking
            different models together. For now, it needs to include the shape information.
        params_hook : List(theano shared variable)
            A list of model parameters (shared theano variables) that you should use when constructing
            this model (instead of initializing your own shared variables).
        outdir : str
            The directory you want outputs (parameters, images, etc.) to save to. If None, nothing will
            be saved.
        input_size : tuple
            Shape of the incoming data: (batch_size, num_channels, input_height, input_width).
        filter_shape : tuple
            (num_filters, num_channels, filter_height, filter_width). This is also the shape of the weights matrix.
        convstride : int
            The distance between the receptive field centers of neighboring units. This is the 'subsample' of theano's
            convolution operation.
        padsize : int
            This is the border_mode for theano's convolution operation.
        group : int
            Not yet supported, used for multi-gpu implementation.
            .. todo:: support multi-gpu
        poolsize : int
            How much to downsample the output.
        poolstride : int
            The stride width for downsampling the output.
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
            callable function, such as cudnn or cuda-convnet if you don't want to use the meta-optimizer.
        mrg : random
            A random number generator that is used when adding noise.
            I recommend using Theano's sandbox.rng_mrg.MRG_RandomStreams.
        """
        super(ConvPoolLayer, self).__init__(**{arg: val for (arg, val) in locals().iteritems() if arg is not 'self'})

        # deal with the inputs coming from inputs_hook - necessary for now to give an input hook
        # inputs_hook is a tuple of (Shape, Input)
        if self.inputs_hook:
            assert len(self.inputs_hook) == 2, "expecting inputs_hook to be tuple of (shape, input)"
            self.input = inputs_hook[1]
        else:
            self.input = T.ftensor4("X")

        self.group = group

        #######################
        # layer configuration #
        #######################
        # activation function!
        self.activation_func = get_activation_function(activation)

        # convolution function!
        if convolution == 'conv2d':
            # using the theano flag optimizer_including=conv_meta will let this conv function optimize itself.
            self.convolution_func = T.nnet.conv2d
        else:
            assert callable(convolution), "Input convolution was not 'conv2d' and was not Callable."
            self.convolution_func = convolution

        # expect image_shape to be bc01!
        self.channel = self.input_size[1]

        self.convstride = convstride
        self.padsize = padsize

        self.poolstride = poolstride
        self.poolsize = poolsize

        # if lib_conv is cudnn, it works only on square images and the grad works only when channel % 16 == 0

        assert self.group in [1, 2], "group argument needs to be 1 or 2 (1 for default conv2d)"

        filter_shape = numpy.asarray(filter_shape)
        self.input_size = numpy.asarray(self.input_size)

        if local_response_normalization:
            lrn_func = cross_channel_normalization_bc01
        else:
            lrn_func = None

        ################################################
        # Params - make sure to deal with params_hook! #
        ################################################
        if self.group == 1:
            if self.params_hook:
                # make sure the params_hook has W and b
                assert len(self.params_hook) == 2, \
                    "Expected 2 params (W and b) for ConvPoolLayer, found {0!s}!".format(len(self.params_hook))
                self.W, self.b = self.params_hook
            else:
                self.W = get_weights(weights_init=weights_init,
                                     shape=filter_shape,
                                     name="W",
                                     rng=mrg,
                                     # if gaussian
                                     mean=weights_mean,
                                     std=weights_std,
                                     # if uniform
                                     interval=weights_interval)

                self.b = get_bias(shape=filter_shape[0], init_values=bias_init, name="b")

            self.params = [self.W, self.b]

        else:
            filter_shape[0] = filter_shape[0] / 2
            filter_shape[1] = filter_shape[1] / 2

            self.input_size[0] = self.input_size[0] / 2
            self.input_size[1] = self.input_size[1] / 2
            if self.params_hook:
                assert len(self.params_hook) == 4, "expected params_hook to have 4 params"
                self.W0, self.W1, self.b0, self.b1 = self.params_hook
            else:
                self.W0 = get_weights_gaussian(shape=filter_shape, name="W0")
                self.W1 = get_weights_gaussian(shape=filter_shape, name="W1")
                self.b0 = get_bias(shape=filter_shape[0], init_values=bias_init, name="b0")
                self.b1 = get_bias(shape=filter_shape[0], init_values=bias_init, name="b1")
            self.params = [self.W0, self.b0, self.W1, self.b1]

        #############################################
        # build appropriate graph for conv. version #
        #############################################
        self.output = self._build_computation_graph()

        # Local Response Normalization (for AlexNet)
        if local_response_normalization and lrn_func is not None:
            self.output = lrn_func(self.output)

        log.debug("convpool layer initialized with shape_in: %s", str(self.input_size))

    def _build_computation_graph(self):
        if self.group == 1:
            conv_out = self.convolution_func(
                img=self.input,
                kerns=self.W,
                subsample=(self.convstride, self.convstride),
                border_mode=(self.padsize, self.padsize)
            )
            conv_out = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        else:
            conv_out0 = self.convolution_func(
                img=self.input[:, :self.channel / 2, :, :],
                kerns=self.W0,
                subsample=(self.convstride, self.convstride),
                border_mode=(self.padsize, self.padsize)
            )
            conv_out0 = conv_out0 + self.b0.dimshuffle('x', 0, 'x', 'x')


            conv_out1 = self.convolution_func(
                img=self.input[:, self.channel / 2:, :, :],
                kerns=self.W1,
                subsample=(self.convstride, self.convstride),
                border_mode=(self.padsize, self.padsize)
            )
            conv_out1 = conv_out1 + self.b1.dimshuffle('x', 0, 'x', 'x')

            conv_out = T.concatenate([conv_out0, conv_out1], axis=1)

        # ReLu by default
        output = self.activation_func(conv_out)

        # Pooling
        if self.poolsize != 1:
            if has_cudnn:
                output = dnn.dnn_pool(output,
                                      ws=(self.poolsize, self.poolsize),
                                      stride=(self.poolstride, self.poolstride))
            else:
                output = downsample.max_pool_2d(output,
                                                ds=(self.poolsize, self.poolsize),
                                                st=(self.poolstride, self.poolstride))
        return output

    def get_inputs(self):
        return [self.input]

    def get_outputs(self):
        return self.output

    def get_params(self):
        return self.params

    def save_args(self, args_file="convpool_config.pkl"):
        super(ConvPoolLayer, self).save_args(args_file)