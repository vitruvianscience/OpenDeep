"""
This module provides a framework for constructing recurrent networks. Recurrent networks have an internal hidden
state that keeps memory over time.
"""
# standard libraries
import logging
# third party libraries
from numpy import prod
from theano import scan
from theano.tensor import (unbroadcast, dot, zeros_like)
from theano.gradient import grad_clip
import theano.sandbox.rng_mrg as RNG_MRG
# internal references
from opendeep.models.model import Model
from opendeep.utils.activation import get_activation_function
from opendeep.utils.decorators import inherit_docs
from opendeep.utils.nnet import (get_weights, get_bias)

log = logging.getLogger(__name__)

@inherit_docs
class RNN(Model):
    """
    Your run-of-the-mill recurrent neural network. This has hidden units that keep track of memory over time.

    Notes
    -----
    Bidirectional and deep hidden layers implemented like in:

    "Towards End-to-End Speech Recognition with Recurrent Neural Networks".
    Alex Graves, Navdeep Jaitly.
    http://www.jmlr.org/proceedings/papers/v32/graves14.pdf

    "Deep Speech: Scaling up end-to-end speech recognition".
    Awni Hannun, Carl Case, Jared Casper, Bryan Catanzaro, Greg Diamos, Erich Elsen,
    Ryan Prenger, Sanjeev Satheesh, Shubho Sengupta, Adam Coates, Andrew Y. Ng.
    http://arxiv.org/pdf/1412.5567v2.pdf

    Stacked layers implemented like in:
    "Gated Feedback Recurrent Neural Networks"
    Junyoung Chung, Caglar Gulcehre, Kyunghyun Cho, Yoshua Bengio
    http://arxiv.org/abs/1502.02367
    """
    def __init__(self, inputs=None, hiddens=None, params=None, outdir='outputs/rnn/',
                 activation='relu',
                 mrg=RNG_MRG.MRG_RandomStreams(1),
                 weights_init='uniform', weights_interval='montreal', weights_mean=0, weights_std=5e-3,
                 bias_init=0.0,
                 r_weights_init='identity', r_weights_interval='montreal', r_weights_mean=0, r_weights_std=5e-3,
                 r_bias_init=0.0,
                 direction='forward',
                 clip_recurrent_grads=False):
        """
        Initialize a simple recurrent network layer.

        Parameters
        ----------
        inputs : List of [tuple(shape, `Theano.TensorType`)]
            The dimensionality of the inputs for this model, and the routing information for the model
            to accept inputs from elsewhere. `inputs` variable are expected to be of the form (timesteps, batch, data).
            `shape` will be a monad tuple representing known
            sizes for each dimension in the `Theano.TensorType`. The length of `shape` should be equal to number of
            dimensions in `Theano.TensorType`, where the shape element is an integer representing the size for its
            dimension, or None if the shape isn't known. For example, if you have a matrix with unknown batch size
            but fixed feature size of 784, `shape` would be: (None, 784). The full form of `inputs` would be:
            [((None, 784), <TensorType(float32, matrix)>)].
        hiddens : int or Tuple of (shape, `Theano.TensorType`)
            Int for the number of hidden units to use, or a tuple of shape, expression to route the starting
            hidden values from elsewhere.
        params : Dict(string_name: theano SharedVariable), optional
            A dictionary of model parameters (shared theano variables) that you should use when constructing
            this model (instead of initializing your own shared variables). This parameter is useful when you want to
            have two versions of the model that use the same parameters - such as siamese networks or pretraining some
            weights.
        outdir : str
            The location to produce outputs from training or running the :class:`RNN`. If None, nothing will be saved.
        activation : str or callable
            The activation function to perform for the hidden layers.
            See opendeep.utils.activation for a list of available activation functions. Alternatively, you can pass
            your own function to be used as long as it is callable.
        mrg : random
            A random number generator that is used when adding noise.
            I recommend using Theano's sandbox.rng_mrg.MRG_RandomStreams.
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
        r_weights_init : str
            Determines the method for initializing recurrent model weights. See opendeep.utils.nnet for options.
        r_weights_interval : str or float
            If Uniform `r_weights_init`, the +- interval to use. See opendeep.utils.nnet for options.
        r_weights_mean : float
            If Gaussian `r_weights_init`, the mean value to use.
        r_weights_std : float
            If Gaussian `r_weights_init`, the standard deviation to use.
        r_bias_init : float
            The initial value to use for the recurrent bias parameter. Most often, the default of 0.0 is preferred.
        direction : str
            The direction this recurrent model should go over its inputs. Can be 'forward', 'backward', or
            'bidirectional'. In the case of 'bidirectional', it will make two passes over the sequence,
            computing two sets of hiddens and adding them together.
        clip_recurrent_grads : False or float, optional
            Whether to clip the gradients for the parameters that unroll over timesteps (such as the weights
            connecting previous hidden states to the current hidden state, and not the weights from current
            input to hiddens). If it is a float, the gradients for the weights will be hard clipped to the range
            `+-clip_recurrent_grads`.

        Raises
        ------
        AssertionError
            When asserting various properties of input parameters. See error messages.
        """
        initial_parameters = locals().copy()
        initial_parameters.pop('self')
        super(RNN, self).__init__(**initial_parameters)

        ##################
        # specifications #
        ##################
        bidirectional = (direction == "bidirectional")
        backward = (direction == "backward")

        ########################
        # activation functions #
        ########################
        # recurrent hidden activation functions!
        self.hidden_activation_func = get_activation_function(activation)

        ##########
        # inputs #
        ##########
        # inputs are expected to have the shape (n_timesteps, batch_size, data)
        if len(self.inputs) > 1:
            raise NotImplementedError("Expected 1 input, found %d. Please merge inputs before passing "
                                      "to the model!" % len(self.inputs))
        # self.inputs is a list of all the input expressions (we enforce only 1, so self.inputs[0] is the input)
        input_shape, self.input = self.inputs[0]
        if isinstance(input_shape, int):
            self.input_size = ((None,) * (self.input.ndim - 1)) + (input_shape,)
        else:
            self.input_size = input_shape
        assert self.input_size is not None, "Need to specify the shape for at least the last dimension of the input!"
        # input is 3D tensor of (timesteps, batch_size, data_dim)
        # if input is 2D tensor, assume it is of the form (timesteps, data_dim) i.e. batch_size is 1. Convert to 3D.
        # if input is > 3D tensor, assume it is of form (timesteps, batch_size, data...) and flatten to 3D.
        if self.input.ndim == 1:
            self.input = unbroadcast(self.input.dimshuffle(0, 'x', 'x'), [1, 2])

        elif self.input.ndim == 2:
            self.input = unbroadcast(self.input.dimshuffle(0, 'x', 1), 1)

        elif self.input.ndim > 3:
            self.input = self.input.flatten(3)
            self.input_size = self.input_size[:2] + (prod(self.input_size[2:]))

        ###########
        # hiddens #
        ###########
        # have only 1 hiddens
        assert len(self.hiddens) == 1, "Expected 1 `hiddens` param, found %d" % len(self.hiddens)
        self.hiddens = self.hiddens[0]
        # if hiddens is an int (hidden size parameter, not routing info)
        h_init = None
        if isinstance(self.hiddens, int):
            self.hidden_size = self.hiddens
        elif isinstance(self.hiddens, tuple):
            hidden_shape, h_init = self.hiddens
            if isinstance(hidden_shape, int):
                self.hidden_size = hidden_shape
            else:
                self.hidden_size = hidden_shape[-1]
        else:
            raise AssertionError("Hiddens need to be an int or tuple of (shape, theano_expression), found %s" %
                                 type(self.hiddens))

        # output shape is going to be 3D with (timesteps, batch_size, hidden_size)
        self.output_size = (None, None, self.hidden_size)

        ##########################################################
        # parameters - make sure to deal with params dict input! #
        ##########################################################
        # input-to-hidden
        W = self.params.get(
            "W",
            get_weights(weights_init=weights_init,
                        shape=(self.input_size[-1], self.hidden_size),
                        name="W",
                        # if gaussian
                        mean=weights_mean,
                        std=weights_std,
                        # if uniform
                        interval=weights_interval)
        )
        # hidden-to-hidden
        U = self.params.get(
            "U",
            get_weights(weights_init=r_weights_init,
                        shape=(self.hidden_size, self.hidden_size),
                        name="U",
                        # if gaussian
                        mean=r_weights_mean,
                        std=r_weights_std,
                        # if uniform
                        interval=r_weights_interval)
        )
        # bidirectional, make hidden-to-hidden in reverse as well
        U_b = None
        if bidirectional:
            U_b = self.params.get(
                "U_b",
                get_weights(weights_init=r_weights_init,
                            shape=(self.hidden_size, self.hidden_size),
                            name="U_b",
                            # if gaussian
                            mean=r_weights_mean,
                            std=r_weights_std,
                            # if uniform
                            interval=r_weights_interval)
            )
        # bias
        b = self.params.get(
            "b",
            get_bias(shape=(self.hidden_size,),
                     name="b",
                     init_values=r_bias_init)
        )
        # clip gradients if we are doing that
        if clip_recurrent_grads:
            clip = abs(clip_recurrent_grads)
            U, U_b = [
                grad_clip(param, -clip, clip) if param is not None
                else None
                for param in [U, U_b]
                ]

        # put all the parameters into our dictionary
        self.params = {"W": W, "U": U, "b": b}
        if bidirectional:
            self.params.update({"U_b": U_b})

        # make h_init the right sized tensor
        if h_init is None:
            h_init = zeros_like(dot(self.input[0], W))

        ###############
        # computation #
        ###############
        # now do the recurrent stuff
        self.hiddens, self.updates = scan(
            fn=self.recurrent_step,
            sequences=[self.input],
            outputs_info=[h_init],
            non_sequences=[W, U, b],
            go_backwards=backward,
            name="rnn_scan",
            strict=True
        )

        # if bidirectional, do the same in reverse!
        if bidirectional:
            hiddens_b, updates_b = scan(
                fn=self.recurrent_step,
                sequences=[self.input],
                outputs_info=[h_init],
                non_sequences=[W, U_b, b],
                go_backwards=not backward,
                name="rnn_scan_back",
                strict=True
            )
            # flip the hiddens to be the right direction
            hiddens_b = hiddens_b[::-1]
            # update stuff
            self.updates.update(updates_b)
            self.hiddens += hiddens_b

        log.info("Initialized an RNN!")

    def recurrent_step(self, x_t, h_tm1, W, U, b):
        """
        Performs one computation step over time.

        Parameters
        ----------
        x_t : tensor
            The current timestep (t) input value.
        h_tm1 : tensor
            The previous timestep (t-1) hidden values.
        W : shared variable
            The input-to-hidden weights matrix to use.
        U : shared variable
            The hidden-to-hidden timestep weights matrix to use (differs when bidirectional).
        b : shared variable
            The hidden bias to use.

        Returns
        -------
        tensor
            h_t the current timestep (t) hidden values.
        """
        h_t = self.hidden_activation_func(
            dot(x_t, W) + dot(h_tm1, U) + b
        )
        return h_t

    ###################
    # Model functions #
    ###################
    def get_inputs(self):
        return [self.input]

    def get_outputs(self):
        return self.hiddens

    def get_updates(self):
        return self.updates

    def get_params(self):
        return self.params
