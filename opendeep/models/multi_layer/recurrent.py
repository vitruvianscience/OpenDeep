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
from opendeep import sharedX, function, theano_allclose
from opendeep.models.model import Model
from opendeep.utils.activation import get_activation_function
from opendeep.utils.cost import get_cost_function
from opendeep.utils.decay import get_decay_function
from opendeep.utils.decorators import inherit_docs
from opendeep.utils.nnet import get_weights, get_bias
from opendeep.utils.noise import get_noise

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
    """
    def __init__(self, inputs_hook=None, hiddens_hook=None, params_hook=None, outdir='outputs/rnn/',
                 input_size=None, hidden_size=None, output_size=None,
                 layers=1,
                 activation='sigmoid', hidden_activation='relu',
                 mrg=RNG_MRG.MRG_RandomStreams(1),
                 weights_init='uniform', weights_interval='montreal', weights_mean=0, weights_std=5e-3,
                 bias_init=0.0,
                 r_weights_init='identity', r_weights_interval='montreal', r_weights_mean=0, r_weights_std=5e-3,
                 r_bias_init=0.0,
                 cost_function='mse', cost_args=None,
                 noise='dropout', noise_level=None, noise_decay=False, noise_decay_amount=.99,
                 direction='forward',
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
        layers : int
            The number of stacked hidden layers to use.
        activation : str or callable
            The nonlinear (or linear) activation to perform after the dot product from hiddens -> output layer.
            This activation function should be appropriate for the output unit types, i.e. 'sigmoid' for binary.
            See opendeep.utils.activation for a list of available activation functions. Alternatively, you can pass
            your own function to be used as long as it is callable.
        hidden_activation : str or callable
            The activation to perform for the hidden layers.
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
        direction : str
            The direction this recurrent model should go over its inputs. Can be 'forward', 'backward', or
            'bidirectional'. In the case of 'bidirectional', it will make two passes over the sequence,
            computing two sets of hiddens and merging them before running through the final decoder.
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
        self.direction = direction
        self.bidirectional = (direction == "bidirectional")
        self.backward = (direction == "backward")
        self.layers = layers
        self.noise = noise

        self.weights_init = weights_init
        self.weights_mean = weights_mean
        self.weights_std = weights_std
        self.weights_interval = weights_interval

        self.r_weights_init = r_weights_init
        self.r_weights_mean = r_weights_mean
        self.r_weights_std = r_weights_std
        self.r_weights_interval = r_weights_interval

        self.bias_init = bias_init
        self.r_bias_init = r_bias_init

        #########################################
        # activation, cost, and noise functions #
        #########################################
        # recurrent hidden activation function!
        self.hidden_activation_func = get_activation_function(hidden_activation)

        # output activation function!
        self.activation_func = get_activation_function(activation)

        # Cost function
        self.cost_function = get_cost_function(cost_function)
        self.cost_args = cost_args or dict()

        # Now deal with noise if we added it:
        if self.noise:
            log.debug('Adding %s noise switch.' % str(noise))
            if noise_level is not None:
                noise_level = sharedX(value=noise_level)
                self.noise_func = get_noise(noise, noise_level=noise_level, mrg=mrg)
            else:
                self.noise_func = get_noise(noise, mrg=mrg)
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
            self.h_init = self.hiddens_hook[1]
            self.hidden_size = self.hiddens_hook[0]
        else:
            # deal with h_init after parameters are made (have to make the same size as hiddens that are computed)
            self.hidden_size = hidden_size

        ##################
        # for generating #
        ##################
        # symbolic scalar for how many recurrent steps to use during generation from the model
        self.n_steps = T.iscalar("generate_n_steps")

        self.output, self.hiddens, self.updates, self.cost, self.params = self.build_computation_graph()

    def build_computation_graph(self):
        """
        Creates the output, hiddens, updates, cost, and parameters for the RNN!

        Returns
        -------
        Output, top-level hiddens, updates, cost, and parameters for the RNN.
        """
        ####################################################
        # parameters - make sure to deal with params_hook! #
        ####################################################
        if self.params_hook is not None:
            # expect at least W_{x_h}, W_{h_h}, W_{h_y}, b_h, b_y -> this is for single-direction RNN.
            assert len(self.params_hook) >= 3*self.layers+2, \
                "Expected at least {0!s} params for rnn, found {1!s}!".format(3*self.layers+2, len(self.params_hook))
            W_x_h = self.params_hook[:self.layers]
            W_h_h = self.params_hook[self.layers:2*self.layers]
            b_h   = self.params_hook[2*self.layers:3*self.layers]
            W_h_y = self.params_hook[3*self.layers]
            b_y   = self.params_hook[3*self.layers+1]
            # now the case for extra parameters dealing with a backward pass in addition to forward (bidirectional)
            if self.bidirectional:
                assert len(self.params_hook) >= 4*self.layers+2, \
                    "Expected at least {0!s} params for bidirectional (merging hiddens) rnn, found {1!s}!".format(
                        4*self.layers+2, len(self.params_hook))
                # if we are merging according to DeepSpeech paper, this is all we need in addition for bidirectional.
                W_h_hb = self.params_hook[3*self.layers+2:4*self.layers+2]
        # otherwise, construct our params
        else:
            # input-to-hidden (and hidden-to-hidden higher layer) weights
            W_x_h = []
            for l in range(self.layers):
                if l > 0:
                    W_x_h.append(get_weights(weights_init=self.weights_init,
                                             shape=(self.hidden_size, self.hidden_size),
                                             name="W_%d_%d" % (l, l+1),
                                             # if gaussian
                                             mean=self.weights_mean,
                                             std=self.weights_std,
                                             # if uniform
                                             interval=self.weights_interval))
                else:
                    W_x_h.append(get_weights(weights_init=self.weights_init,
                                             shape=(self.input_size, self.hidden_size),
                                             name="W_%d_%d" % (l, l+1),
                                             # if gaussian
                                             mean=self.weights_mean,
                                             std=self.weights_std,
                                             # if uniform
                                             interval=self.weights_interval))
            # hidden-to-hidden same layer weights
            W_h_h = [get_weights(weights_init=self.r_weights_init,
                                 shape=(self.hidden_size, self.hidden_size),
                                 name="W_%d_%d" % (l+1, l+1),
                                 # if gaussian
                                 mean=self.r_weights_mean,
                                 std=self.r_weights_std,
                                 # if uniform
                                 interval=self.r_weights_interval)
                     for l in range(self.layers)]
            # hidden-to-output weights
            W_h_y = get_weights(weights_init=self.weights_init,
                                shape=(self.hidden_size, self.output_size),
                                name="W_h_y",
                                # if gaussian
                                mean=self.weights_mean,
                                std=self.weights_std,
                                # if uniform
                                interval=self.weights_interval)
            # hidden bias for each layer
            b_h = [get_bias(shape=(self.hidden_size,),
                            name="b_h_%d" % (l+1),
                            init_values=self.r_bias_init)
                   for l in range(self.layers)]
            # output bias
            b_y = get_bias(shape=(self.output_size,),
                           name="b_y",
                           init_values=self.bias_init)
            # extra parameters necessary for second backward pass on hiddens if this is bidirectional
            if self.bidirectional:
                # hidden-to-hidden same layer backward weights.
                W_h_hb = [get_weights(weights_init=self.r_weights_init,
                                      shape=(self.hidden_size, self.hidden_size),
                                      name="W_%d_%db" % (l+1, l+1),
                                      # if gaussian
                                      mean=self.r_weights_mean,
                                      std=self.r_weights_std,
                                      # if uniform
                                      interval=self.r_weights_interval)
                          for l in range(self.layers)]

        # if we are clipping the recurrent gradients, use theano.gradient.grad_clip(x, lower_bound, upper_bound).

        # put all the parameters into our list, and make sure it is in the same order as when we try to load
        # them from a params_hook!!!
        params = W_x_h + W_h_h + b_h + [W_h_y] + [b_y]
        if self.bidirectional:
            params += W_h_hb

        # make h_init the right sized tensor
        if not self.hiddens_hook:
            self.h_init = T.zeros_like(T.dot(self.input[0], W_x_h[0]))

        ###############
        # computation #
        ###############
        hiddens = self.input
        updates = dict()
        # vanilla case! there will be only 1 hidden layer for each depth layer.
        for layer in range(self.layers):
            log.debug("Updating hidden layer %d" % (layer+1))
            # normal case - either forward or just backward!
            hiddens_new, updates = theano.scan(
                fn=self.recurrent_step,
                sequences=hiddens,
                outputs_info=self.h_init,
                non_sequences=[W_x_h[layer], W_h_h[layer], b_h[layer]],
                go_backwards=self.backward,
                name="rnn_scan_normal_%d" % layer,
                strict=True
            )
            updates.update(updates)

            # bidirectional case - need to add a backward sequential pass to compute new hiddens!
            if self.bidirectional:
                # now do the opposite direction for the scan!
                hiddens_opposite, updates_opposite = theano.scan(
                    fn=self.recurrent_step,
                    sequences=hiddens,
                    outputs_info=self.h_init,
                    non_sequences=[W_x_h[layer], W_h_hb[layer], b_h[layer]],
                    go_backwards=(not self.backward),
                    name="rnn_scan_backward_%d" % layer,
                    strict=True
                )
                updates.update(updates_opposite)
                hiddens_new = hiddens_new + hiddens_opposite

            # replace the hiddens with the newly computed hiddens (and add noise)!
            hiddens = hiddens_new
            # add noise (like dropout) if we wanted it!
            if self.noise:
                self.hiddens = T.switch(self.noise_switch,
                                        self.noise_func(input=hiddens),
                                        hiddens)

        # now compute the outputs from the leftover (top level) hiddens
        output = self.activation_func(
            T.dot(hiddens, W_h_y) + b_y
        )

        # now to define the cost of the model - use the cost function to compare our output with the target value.
        cost = self.cost_function(output=output, target=self.target, **self.cost_args)

        log.info("Initialized a %s RNN!" % self.direction)
        return output, hiddens, updates, cost, params

    def recurrent_step(self, x_t, h_tm1, W_x_h, W_h_h, b_h):
        """
        Performs one computation step over time.

        Parameters
        ----------
        x_t : tensor
            The current timestep (t) input value.
        h_tm1 : tensor
            The previous timestep (t-1) hidden values.
        W_x_h : shared variable
            The input-to-hidden weights matrix to use.
        W_h_h : shared variable
            The hidden-to-hidden timestep weights matrix to use (differs when bidirectional).
        b_h : shared variable
            The hidden bias to use (differs when bidirectional).

        Returns
        -------
        tensor
            h_t the current timestep (t) hidden values.
        """
        h_t = self.hidden_activation_func(
            T.dot(x_t, W_x_h) + T.dot(h_tm1, W_h_h) + b_h
        )
        return h_t

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
            return super(RNN, self).get_decay_params()

    def get_noise_switch(self):
        if hasattr(self, 'noise_switch'):
            return [self.noise_switch]
        else:
            return super(RNN, self).get_noise_switch()

    def get_params(self):
        return self.params

    def save_args(self, args_file="rnn_config.pkl"):
        super(RNN, self).save_args(args_file)