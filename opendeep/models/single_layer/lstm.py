"""
This module provides a framework for constructing recurrent networks. Recurrent networks have an internal hidden
state that keeps memory over time.
"""
__authors__ = ["Markus Beissinger", "Skylar Payne"]
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger", "Skylar Payne"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
# third party libraries
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG
# internal references
from opendeep.utils.constructors import sharedX
from opendeep.models.model import Model
from opendeep.utils.activation import get_activation_function
from opendeep.utils.cost import get_cost_function
from opendeep.utils.decay import get_decay_function
from opendeep.utils.decorators import inherit_docs
from opendeep.utils.nnet import get_weights, get_bias
from opendeep.utils.noise import get_noise

log = logging.getLogger(__name__)

@inherit_docs
class LSTM(Model):
    """
    Your normal lstm.

    Implemented from:
    "Gated Feedback Recurrent Neural Networks"
    Junyoung Chung, Caglar Gulcehre, Kyunghyun Cho, Yoshua Bengio
    http://arxiv.org/pdf/1502.02367v3.pdf
    """
    def __init__(self, inputs_hook=None, hiddens_hook=None, params_hook=None, outdir='outputs/rnn/',
                 input_size=None, hidden_size=None, output_size=None,
                 layers=1,
                 activation='sigmoid', hidden_activation='relu', inner_hidden_activation='sigmoid',
                 mrg=RNG_MRG.MRG_RandomStreams(1),
                 weights_init='uniform', weights_interval='montreal', weights_mean=0, weights_std=5e-3,
                 bias_init=0.0,
                 r_weights_init='identity', r_weights_interval='montreal', r_weights_mean=0, r_weights_std=5e-3,
                 r_bias_init=0.0,
                 cost_function='mse', cost_args=None,
                 noise='dropout', noise_level=None, noise_decay=False, noise_decay_amount=.99,
                 forward=True,
                 clip_recurrent_grads=False):
        """
        Initialize a simple recurrent network.

        Parameters
        ----------
        inputs_hook : Tuple of (shape, variable)
            Routing information for the model to accept inputs from elsewhere. This is used for linking
            different models together (e.g. setting the Softmax model's input layer to the DAE's hidden layer gives a
            newly supervised classification model). For now, it needs to include the shape information (normally the
            dimensionality of the input i.e. n_in).
        hiddens_hook : Tuple of (shape, variable)
            Routing information for the model to accept its hidden representation from elsewhere. For recurrent nets,
            this will be the initial starting value for hidden layers.
        params_hook : List(theano shared variable)
            A list of model parameters (shared theano variables) that you should use when constructing
            this model (instead of initializing your own shared variables). This parameter is useful when you want to
            have two versions of the model that use the same parameters.
        outdir : str
            The location to produce outputs from training or running the :class:`RNN`. If None, nothing will be saved.
        input_size : int
            The size (dimensionality) of the input. If shape is provided in `inputs_hook`, this is optional.
        hidden_size : int
            The size (dimensionality) of the hidden layers. If shape is provided in `hiddens_hook`, this is optional.
        output_size : int
            The size (dimensionality) of the output.
        activation : str or callable
            The nonlinear (or linear) activation to perform after the dot product from hiddens -> output layer.
            This activation function should be appropriate for the output unit types, i.e. 'sigmoid' for binary.
            See opendeep.utils.activation for a list of available activation functions. Alternatively, you can pass
            your own function to be used as long as it is callable.
        hidden_activation : str or callable
            The activation to perform for the hidden units.
            See opendeep.utils.activation for a list of available activation functions. Alternatively, you can pass
            your own function to be used as long as it is callable.
        inner_hidden_activation : str or callable
            The activation to perform for the hidden gates.
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
        cost_function : str or callable
            The function to use when calculating the output cost of the model.
            See opendeep.utils.cost for options. You can also specify your own function, which needs to be callable.
        cost_args : dict
            Any additional named keyword arguments to pass to the specified `cost_function`.
        noise : str
            What type of noise to use for the hidden layers and outputs. See opendeep.utils.noise
            for options. This should be appropriate for the unit activation, i.e. Gaussian for tanh or other
            real-valued activations, etc.
        noise_level : float
            The amount of noise to use for the noise function specified by `hidden_noise`. This could be the
            standard deviation for gaussian noise, the interval for uniform noise, the dropout amount, etc.
        noise_decay : str or False
            Whether to use `noise` scheduling (decay `noise_level` during the course of training),
            and if so, the string input specifies what type of decay to use. See opendeep.utils.decay for options.
            Noise decay (known as noise scheduling) effectively helps the model learn larger variance features first,
            and then smaller ones later (almost as a kind of curriculum learning). May help it converge faster.
        noise_decay_amount : float
            The amount to reduce the `noise_level` after each training epoch based on the decay function specified
            in `noise_decay`.
        forward : bool
            The direction this recurrent model should go over its inputs. True means forward, False mean backward.
        clip_recurrent_grads : False or float, optional
            Whether to clip the gradients for the parameters that unroll over timesteps (such as the weights
            connecting previous hidden states to the current hidden state, and not the weights from current
            input to hiddens). If it is a float, the gradients for the weights will be hard clipped to the range
            `+-clip_recurrent_grads`.
        """
        initial_parameters = locals().copy()
        initial_parameters.pop('self')
        super(LSTM, self).__init__(**initial_parameters)

        ##################
        # specifications #
        ##################

        #########################################
        # activation, cost, and noise functions #
        #########################################
        # recurrent hidden activation function!
        self.hidden_activation_func = get_activation_function(hidden_activation)
        self.inner_hidden_activation_func = get_activation_function(inner_hidden_activation)

        # output activation function!
        activation_func = get_activation_function(activation)

        # Cost function
        cost_function = get_cost_function(cost_function)
        cost_args = cost_args or dict()

        # Now deal with noise if we added it:
        if noise:
            log.debug('Adding %s noise switch.' % str(noise))
            if noise_level is not None:
                noise_level = sharedX(value=noise_level)
                noise_func = get_noise(noise, noise_level=noise_level, mrg=mrg)
            else:
                noise_func = get_noise(noise, mrg=mrg)
            # apply the noise as a switch!
            # default to apply noise. this is for the cost and gradient functions to be computed later
            # (not sure if the above statement is accurate such that gradient depends on initial value of switch)
            self.noise_switch = sharedX(value=1, name="basiclayer_noise_switch")

            # noise scheduling
            if noise_decay and noise_level is not None:
                self.noise_schedule = get_decay_function(noise_decay,
                                                         noise_level,
                                                         noise_level.get_value(),
                                                         noise_decay_amount)

        ###############
        # inputs hook #
        ###############
        # grab info from the inputs_hook
        # in the case of an inputs_hook, recurrent will always work with the leading tensor dimension
        # being the temporal dimension.
        # input is 3D tensor of (timesteps, batch_size, data_dim)
        # if input is 2D tensor, assume it is of the form (timesteps, data_dim) i.e. batch_size is 1. Convert to 3D.
        # if input is > 3D tensor, assume it is of form (timesteps, batch_size, data...) and flatten to 3D.
        if self.inputs_hook is not None:
            self.input = self.inputs_hook[1]

            if self.input.ndim == 1:
                self.input = self.input.dimshuffle(0, 'x', 'x')
                self.input_size = 1

            elif self.input.ndim == 2:
                self.input = self.input.dimshuffle(0, 'x', 1)

            elif self.input.ndim > 3:
                self.input = self.input.flatten(3)
                self.input_size = sum(self.input_size)
            else:
                raise NotImplementedError("Recurrent input with %d dimensions not supported!" % self.input.ndim)
        else:
            # Assume input coming from optimizer is (batches, timesteps, data)
            # so, we need to reshape to (timesteps, batches, data)
            xs = T.tensor3("Xs")
            xs = xs.dimshuffle(1, 0, 2)
            self.input = xs

        # The target outputs for supervised training - in the form of (batches, timesteps, output) which is
        # the same dimension ordering as the expected input from optimizer.
        # therefore, we need to swap it like we did to input xs.
        ys = T.tensor3("Ys")
        ys = ys.dimshuffle(1, 0, 2)
        self.target = ys

        ################
        # hiddens hook #
        ################
        # set an initial value for the recurrent hiddens from hook
        if self.hiddens_hook is not None:
            h_init = self.hiddens_hook[1]
            self.hidden_size = self.hiddens_hook[0]
        else:
            # deal with h_init after parameters are made (have to make the same size as hiddens that are computed)
            self.hidden_size = hidden_size

        ##################
        # for generating #
        ##################
        # symbolic scalar for how many recurrent steps to use during generation from the model
        self.n_steps = T.iscalar("generate_n_steps")

        ####################################################
        # parameters - make sure to deal with params_hook! #
        ####################################################
        if self.params_hook is not None:
            (W_x_c, W_x_i, W_x_f, W_x_o,
             U_h_c, U_h_i, U_h_f, U_h_o,
             W_h_y, b_c, b_i, b_f, b_o,
             b_y) = self.params_hook
        # otherwise, construct our params
        else:
            # all input-to-hidden weights
            W_x_c, W_x_i, W_x_f, W_x_o = [
                get_weights(weights_init=weights_init,
                            shape=(self.input_size, self.hidden_size),
                            name="W_x_%s" % sub,
                            # if gaussian
                            mean=weights_mean,
                            std=weights_std,
                            # if uniform
                            interval=weights_interval)
                for sub in ['c', 'i', 'f', 'o']
            ]
            # all hidden-to-hidden weights
            U_h_c, U_h_i, U_h_f, U_h_o = [
                get_weights(weights_init=r_weights_init,
                            shape=(self.hidden_size, self.hidden_size),
                            name="U_h_%s" % sub,
                            # if gaussian
                            mean=r_weights_mean,
                            std=r_weights_std,
                            # if uniform
                            interval=r_weights_interval)
                for sub in ['c', 'i', 'f', 'o']
            ]
            # hidden-to-output weights
            W_h_y = get_weights(weights_init=weights_init,
                                shape=(self.input_size, self.hidden_size),
                                name="W_h_y",
                                # if gaussian
                                mean=weights_mean,
                                std=weights_std,
                                # if uniform
                                interval=weights_interval)
            # biases
            b_c, b_i, b_f, b_o = [
                get_bias(shape=(self.hidden_size,),
                         name="b_%s" % sub,
                         init_values=r_bias_init)
                for sub in ['c', 'i', 'f', 'o']
            ]
            # output bias
            b_y = get_bias(shape=(self.output_size,),
                           name="b_y",
                           init_values=bias_init)
            # clip gradients if we are doing that
            recurrent_params = [U_h_c, U_h_i, U_h_f, U_h_o]
            if clip_recurrent_grads:
                clip = abs(clip_recurrent_grads)
                U_h_c, U_h_i, U_h_f, U_h_o = [theano.gradient.grad_clip(p, -clip, clip) for p in recurrent_params]

        # put all the parameters into our list, and make sure it is in the same order as when we try to load
        # them from a params_hook!!!
        params = [W_x_c, W_x_i, W_x_f, W_x_o,
                  U_h_c, U_h_i, U_h_f, U_h_o,
                  W_h_y, b_c, b_i, b_f, b_o,
                  b_y]

        # make h_init the right sized tensor
        if not self.hiddens_hook:
            h_init = T.zeros_like(T.dot(self.input[0], W_x_c))

        c_init = T.zeros_like(T.dot(self.input[0], W_x_c))

        ###############
        # computation #
        ###############
        # move some computation outside of scan to speed it up!
        x_c = T.dot(self.input, W_x_c) + b_c
        x_i = T.dot(self.input, W_x_i) + b_i
        x_f = T.dot(self.input, W_x_f) + b_f
        x_o = T.dot(self.input, W_x_o) + b_o

        # now do the recurrent stuff
        self.hiddens, memories, self.updates = theano.scan(
            fn=self.recurrent_step,
            sequences=[x_c, x_i, x_f, x_o],
            outputs_info=[h_init, c_init],
            non_sequences=[U_h_c, U_h_i, U_h_f, U_h_o],
            go_backwards=not forward,
            name="lstm_scan",
            strict=True
        )

        # add noise (like dropout) if we wanted it!
        if noise:
            self.hiddens = T.switch(self.noise_switch,
                                    self.noise_func(input=self.hiddens),
                                    self.hiddens)

        # now compute the outputs from the leftover (top level) hiddens
        output = self.activation_func(
            T.dot(self.hiddens, W_h_y) + b_y
        )

        # now to define the cost of the model - use the cost function to compare our output with the target value.
        cost = self.cost_function(output=output, target=self.target, **self.cost_args)

        log.info("Initialized an LSTM!")

    def recurrent_step(self, x_c_t, x_i_t, x_f_t, x_o_t, h_tm1, c_tm1, U_h_c, U_h_i, U_h_f, U_h_o):
        """
        Performs one computation step over time.
        """
        # new memory content c_tilde
        c_tilde = self.hidden_activation_func(
            x_c_t + T.dot(h_tm1, U_h_c)
        )
        # input gate
        i_t = self.inner_hidden_activation_func(
            x_i_t + T.dot(h_tm1, U_h_i)
        )
        # forget gate
        f_t = self.inner_hidden_activation_func(
            x_f_t + T.dot(h_tm1, U_h_f)
        )
        # new memory content
        c_t = f_t*c_tm1 + i_t*c_tilde
        # output gate
        o_t = self.inner_hidden_activation_func(
            x_o_t + T.dot(h_tm1, U_h_o)
        )
        # new hiddens
        h_t = o_t*self.hidden_activation_func(c_t)
        # return the hiddens and memory content
        return h_t, c_t

    ###################
    # Model functions #
    ###################
    def get_inputs(self):
        return [self.input]

    def get_hiddens(self):
        return self.hiddens

    def get_outputs(self):
        return self.output

    def get_targets(self):
        return [self.target]

    def get_train_cost(self):
        return self.cost

    def get_updates(self):
        return self.updates

    def get_monitors(self):
        # TODO: calculate monitors we might care about.
        return []

    def get_decay_params(self):
        if hasattr(self, 'noise_schedule'):
            # noise scheduling
            return [self.noise_schedule]
        else:
            return super(LSTM, self).get_decay_params()

    def get_noise_switch(self):
        if hasattr(self, 'noise_switch'):
            return [self.noise_switch]
        else:
            return super(LSTM, self).get_noise_switch()

    def get_params(self):
        return self.params

    def save_args(self, args_file="lstm_config.pkl"):
        super(LSTM, self).save_args(args_file)