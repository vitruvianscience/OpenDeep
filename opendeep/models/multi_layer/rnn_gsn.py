"""
This module provides the RNN-GSN: an unsupervised, generative recurrent model.
.. todo:: <link to my paper when it is on arxiv>
"""

__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
import os
import cPickle as pickle
# third party libraries
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG
# internal references
from opendeep import function
from opendeep.models.model import Model
from opendeep.models.multi_layer.generative_stochastic_network import GSN
from opendeep.utils.decorators import inherit_docs
from opendeep.utils.nnet import get_weights, get_bias
from opendeep.utils.activation import get_activation_function, is_binary
from opendeep.utils import file_ops
from opendeep.utils.cost import get_cost_function

log = logging.getLogger(__name__)


@inherit_docs
class RNN_GSN(Model):
    """
    This gives the Recurrent Generative Stochastic Network (RNN-GSN). It is a generative, unsupervised model
    for sequence representation.
    """
    def __init__(self, inputs_hook=None, hiddens_hook=None, params_hook=None, outdir=None,
                 input_size=None, hidden_size=None,
                 layers=2, walkbacks=4,
                 visible_activation='sigmoid', hidden_activation='tanh',
                 input_sampling=True, mrg=RNG_MRG.MRG_RandomStreams(1),
                 tied_weights=True,
                 weights_init='uniform', weights_interval='montreal', weights_mean=0, weights_std=5e-3,
                 bias_init=0,
                 cost_function='binary_crossentropy', cost_args=None,
                 add_noise=True, noiseless_h1=True,
                 hidden_noise='gaussian', hidden_noise_level=2, input_noise='salt_and_pepper', input_noise_level=0.4,
                 noise_decay='exponential', noise_annealing=1,
                 image_width=None, image_height=None,
                 rnn_hidden_size=None, rnn_hidden_activation='rectifier',
                 rnn_weights_init='identity',
                 rnn_weights_mean=0, rnn_weights_std=5e-3, rnn_weights_interval='montreal',
                 rnn_bias_init=0,
                 generate_n_steps=200):
        """
        Initialize an RNN-GSN.

        Parameters
        ----------
        inputs_hook : Tuple of (shape, variable)
            Routing information for the model to accept inputs from elsewhere. This is used for linking
            different models together (e.g. setting the Softmax model's input layer to the DAE's hidden layer gives a
            newly supervised classification model). For now, it needs to include the shape information (normally the
            dimensionality of the input i.e. n_in).
        hiddens_hook : Tuple of (shape, variable)
            Routing information for the model to accept its hidden representation from elsewhere.
            This is used for linking different models together (e.g. setting the DAE model's hidden layers to the RNN's
            output layer gives a generative recurrent model.) For now, it needs to include the shape
            information (normally the dimensionality of the hiddens i.e. n_hidden).
        params_hook : List(theano shared variable)
            A list of model parameters (shared theano variables) that you should use when constructing
            this model (instead of initializing your own shared variables). This parameter is useful when you want to
            have two versions of the model that use the same parameters - such as a training model with dropout applied
            to layers and one without for testing, where the parameters are shared between the two.
        outdir : str
            The directory you want outputs (parameters, images, etc.) to save to. If None, nothing will
            be saved.
        input_size : int
            The size (dimensionality) of the input to the DAE. If shape is provided in `inputs_hook`, this is optional.
            The :class:`Model` requires an `output_size`, which gets set to this value because the DAE is an
            unsupervised model. The output is a reconstruction of the input.
        hidden_size : int
            The size (dimensionality) of the hidden layer for the DAE. Generally, you want it to be larger than
            `input_size`, which is known as *overcomplete*.
        layers : int
            The number of hidden layers to use.
        walkbacks : int
            The number of walkbacks to perform (the variable K in Bengio's paper above). A walkback is a Gibbs sample
            from the DAE, which means the model generates inputs in sequence, where each generated input is compared
            to the original input to create the reconstruction cost for training. For running the model, the very last
            generated input in the Gibbs chain is used as the output.
        visible_activation : str or callable
            The nonlinear (or linear) visible activation to perform after the dot product from hiddens -> visible layer.
            This activation function should be appropriate for the input unit types, i.e. 'sigmoid' for binary inputs.
            See opendeep.utils.activation for a list of available activation functions. Alternatively, you can pass
            your own function to be used as long as it is callable.
        hidden_activation : str or callable
            The nonlinear (or linear) hidden activation to perform after the dot product from visible -> hiddens layer.
            See opendeep.utils.activation for a list of available activation functions. Alternatively, you can pass
            your own function to be used as long as it is callable.
        input_sampling : bool
            During walkbacks, whether to sample from the generated input to create a new starting point for the next
            walkback (next step in the Gibbs chain). This generally makes walkbacks more effective by making the
            process more stochastic - more likely to find spurious modes in the model's representation.
        mrg : random
            A random number generator that is used when adding noise into the network and for sampling from the input.
            I recommend using Theano's sandbox.rng_mrg.MRG_RandomStreams.
        tied_weights : bool
            DAE has two weight matrices - W from input -> hiddens and V from hiddens -> input. This boolean
            determines if V = W.T, which 'ties' V to W and reduces the number of parameters necessary during training.
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
        cost_function : str or callable
            The function to use when calculating the reconstruction cost of the model. This should be appropriate
            for the type of input, i.e. use 'binary_crossentropy' for binary inputs, or 'mse' for real-valued inputs.
            See opendeep.utils.cost for options. You can also specify your own function, which needs to be callable.
        cost_args : dict
            Any additional named keyword arguments to pass to the specified `cost_function`.
        add_noise : bool
            Whether to add noise (corrupt) the input before passing it through the computation graph during training.
            This should most likely be set to the default of True, because this is a *denoising* autoencoder after all.
        noiseless_h1 : bool
            Whether to not add noise (corrupt) the hidden layer during computation.
        hidden_noise : str
            What type of noise to use for corrupting the hidden layer (if not `noiseless_h1`). See opendeep.utils.noise
            for options. This should be appropriate for the hidden unit activation, i.e. Gaussian for tanh or other
            real-valued activations, etc.
        hidden_noise_level : float
            The amount of noise to use for the noise function specified by `hidden_noise`. This could be the
            standard deviation for gaussian noise, the interval for uniform noise, the dropout amount, etc.
        input_noise : str
            What type of noise to use for corrupting the input before computation (if `add_noise`).
            See opendeep.utils.noise for options. This should be appropriate for the input units, i.e. salt-and-pepper
            for binary units, etc.
        input_noise_level : float
            The amount of noise used to corrupt the input. This could be the masking probability for salt-and-pepper,
            standard deviation for Gaussian, interval for Uniform, etc.
        noise_decay : str or False
            Whether to use `input_noise` scheduling (decay `input_noise_level` during the course of training),
            and if so, the string input specifies what type of decay to use. See opendeep.utils.decay for options.
            Noise decay (known as noise scheduling) effectively helps the DAE learn larger variance features first,
            and then smaller ones later (almost as a kind of curriculum learning). May help it converge faster.
        noise_annealing : float
            The amount to reduce the `input_noise_level` after each training epoch based on the decay function specified
            in `noise_decay`.
        image_width : int
            If the input should be represented as an image, the width of the input image. If not specified, it will be
            close to the square factor of the `input_size`.
        image_height : int
            If the input should be represented as an image, the height of the input image. If not specified, it will be
            close to the square factor of the `input_size`.
        rnn_hidden_size : int
            The number of hidden units (dimensionality) to use in the recurrent layer.
        rnn_hidden_activation : str or Callable
            The activation function to apply to recurrent units. See opendeep.utils.activation for options.
        rnn_weights_init : str
            Determines the method for initializing recurrent weights. See opendeep.utils.nnet for options. 'Identity'
            works well with 'rectifier' `rnn_hidden_activation`.
        rnn_weights_mean : float
            If Gaussian `rnn_weights_init`, the mean value to use.
        rnn_weights_std : float
            If Gaussian `rnn_weights_init`, the standard deviation to use.
        rnn_weights_interval : str or float
            If Uniform `rnn_weights_init`, the +- interval to use. See opendeep.utils.nnet for options.
        rnn_bias_init : float
            The initial value to use for the recurrent bias parameter. Most often, the default of 0.0 is preferred.
        generate_n_steps : int
            When generating from the model, how many steps to generate.
        """
        initial_parameters = locals().copy()
        initial_parameters.pop('self')
        super(RNN_GSN, self).__init__(**initial_parameters)

        ##################
        # specifications #
        ##################
        self.layers = layers
        self.walkbacks = walkbacks
        self.input_sampling = input_sampling
        self.mrg = mrg
        self.tied_weights = tied_weights
        self.noise_decay = noise_decay
        self.noise_annealing = noise_annealing
        self.add_noise = add_noise
        self.noiseless_h1 = noiseless_h1
        self.hidden_noise = hidden_noise
        self.hidden_noise_level = hidden_noise_level
        self.input_noise = input_noise
        self.input_noise_level = input_noise_level
        self.image_width = image_width
        self.image_height = image_height


        # grab info from the inputs_hook, hiddens_hook, or from parameters
        if self.inputs_hook is not None:  # inputs_hook is a tuple of (Shape, Input)
            raise NotImplementedError("Inputs_hook not implemented yet for RNN-GSN")
        else:
            # make the input a symbolic matrix - a sequence of inputs
            self.input = T.matrix('Xs')

        # set an initial value for the recurrent hiddens
        self.u0 = T.zeros((rnn_hidden_size,))

        # make a symbolic vector for the initial recurrent hiddens value to use during generation for the model
        self.generate_u0 = T.vector("generate_u0")

        # either grab the hidden's desired size from the parameter directly, or copy n_in
        self.hidden_size = hidden_size or self.input_size

        # deal with hiddens_hook
        if self.hiddens_hook is not None:
            raise NotImplementedError("Hiddens_hook not implemented yet for RNN-GSN")

        # other specifications
        # visible activation function!
        self.visible_activation_func = get_activation_function(visible_activation)

        # make sure the sampling functions are appropriate for the activation functions.
        if is_binary(self.visible_activation_func):
            self.visible_sampling = mrg.binomial
        else:
            # TODO: implement non-binary activation
            log.error("Non-binary visible activation not supported yet!")
            raise NotImplementedError("Non-binary visible activation not supported yet!")

        # hidden activation function!
        self.hidden_activation_func = get_activation_function(hidden_activation)

        # recurrent hidden activation function!
        self.rnn_hidden_activation_func = get_activation_function(rnn_hidden_activation)

        # Cost function
        self.cost_function = get_cost_function(cost_function)
        self.cost_args = cost_args

        # symbolic scalar for how many recurrent steps to use during generation from the model
        self.n_steps = T.iscalar("generate_n_steps")

        # determine the sizes of each layer in a list.
        # layer sizes, from h0 to hK (h0 is the visible layer)
        self.layer_sizes = [self.input_size] + [self.hidden_size] * self.layers

        ####################################################
        # parameters - make sure to deal with params_hook! #
        ####################################################
        if self.params_hook is not None:
            # if tied weights, expect (layers*2 + 1) params for GSN and (int(layers+1)/int(2) + 3) for RNN
            if self.tied_weights:
                expected_num = (2*self.layers + 1) + (int(self.layers+1)/2 + 3)
                assert len(self.params_hook) == expected_num, \
                    "Tied weights: expected {0!s} params, found {1!s}!".format(expected_num, len(self.params_hook))
                gsn_len = (2*self.layers + 1)
                self.weights_list = self.params_hook[:self.layers]
                self.bias_list = self.params_hook[self.layers:gsn_len]

            # if untied weights, expect layers*3 + 1 params
            else:
                expected_num = (3*self.layers + 1) + (int(self.layers + 1)/2 + 3)
                assert len(self.params_hook) == expected_num, \
                    "Untied weights: expected {0!s} params, found {1!s}!".format(expected_num, len(self.params_hook))
                gsn_len = (3*self.layers + 1)
                self.weights_list = self.params_hook[:2*self.layers]
                self.bias_list = self.params_hook[2*self.layers:gsn_len]

            rnn_len = gsn_len + int(self.layers + 1) / 2
            self.recurrent_to_gsn_weights_list = self.params_hook[gsn_len:rnn_len]
            self.W_u_u = self.params_hook[rnn_len:rnn_len + 1]
            self.W_x_u = self.params_hook[rnn_len + 1:rnn_len + 2]
            self.recurrent_bias = self.params_hook[rnn_len + 2:rnn_len + 3]

        # otherwise, construct our params
        else:
            # initialize a list of weights and biases based on layer_sizes for the GSN
            self.weights_list = [get_weights(weights_init=weights_init,
                                             shape=(self.layer_sizes[i], self.layer_sizes[i + 1]),
                                             name="W_{0!s}_{1!s}".format(i, i + 1),
                                             # if gaussian
                                             mean=weights_mean,
                                             std=weights_std,
                                             # if uniform
                                             interval=weights_interval)
                                 for i in range(self.layers)]
            # add more weights if we aren't tying weights between layers (need to add for higher-lower layers now)
            if not self.tied_weights:
                self.weights_list.extend(
                    [get_weights(weights_init=weights_init,
                                 shape=(self.layer_sizes[i + 1], self.layer_sizes[i]),
                                 name="W_{0!s}_{1!s}".format(i + 1, i),
                                 # if gaussian
                                 mean=weights_mean,
                                 std=weights_std,
                                 # if uniform
                                 interval=weights_interval)
                     for i in reversed(range(self.layers))]
                )
            # initialize each layer bias to 0's.
            self.bias_list = [get_bias(shape=(self.layer_sizes[i],),
                                       name='b_' + str(i),
                                       init_values=bias_init)
                              for i in range(self.layers + 1)]

            self.recurrent_to_gsn_weights_list = [
                get_weights(weights_init=rnn_weights_init,
                            shape=(rnn_hidden_size, self.layer_sizes[layer]),
                            name="W_u_h{0!s}".format(layer),
                            # if gaussian
                            mean=rnn_weights_mean,
                            std=rnn_weights_std,
                            # if uniform
                            interval=rnn_weights_interval)
                for layer in range(self.layers + 1) if layer % 2 != 0
            ]
            self.W_u_u = get_weights(weights_init=rnn_weights_init,
                                     shape=(rnn_hidden_size, rnn_hidden_size),
                                     name="W_u_u",
                                     # if gaussian
                                     mean=rnn_weights_mean,
                                     std=rnn_weights_std,
                                     #if uniform
                                     interval=rnn_weights_interval)
            self.W_x_u = get_weights(weights_init=rnn_weights_init,
                                     shape=(self.input_size, rnn_hidden_size),
                                     name="W_x_u",
                                     # if gaussian
                                     mean=rnn_weights_mean,
                                     std=rnn_weights_std,
                                     # if uniform
                                     interval=rnn_weights_interval)
            self.recurrent_bias = get_bias(shape=(rnn_hidden_size,),
                                           name="b_u",
                                           init_values=rnn_bias_init)

        # build the params of the model into a list
        self.gsn_params = self.weights_list + self.bias_list
        self.params = self.gsn_params + \
                      self.recurrent_to_gsn_weights_list + \
                      [self.W_u_u, self.W_x_u, self.recurrent_bias]
        log.debug("rnn-gsn params: %s", str(self.params))

        # Create the RNN-GSN graph!
        self.x_sample, self.cost, self.monitors, self.updates_train, self.x_ts, self.updates_generate, self.u_t = \
            self._build_rnngsn()

        log.info("Initialized an RNN-GSN!")

    def _build_rnngsn(self):
        """
        Creates the updates and other return variables for the computation graph.

        Returns
        -------
        List
            the sample at the end of the computation graph, the train cost function, the train monitors,
            the computation updates, the generated visible list, the generated computation updates, the ending
            recurrent states
        """
        # For training, the deterministic recurrence is used to compute all the
        # {h_t, 1 <= t <= T} given Xs. Conditional GSNs can then be trained
        # in batches using those parameters.
        (u, h_ts), updates_train = theano.scan(fn=lambda x_t, u_tm1: self.recurrent_step(x_t, u_tm1),
                                               sequences=self.input,
                                               outputs_info=[self.u0, None],
                                               name="rnngsn_computation_scan")

        h_list = [T.zeros_like(self.input)]
        for layer, w in enumerate(self.weights_list[:self.layers]):
            if layer % 2 != 0:
                h_list.append(T.zeros_like(T.dot(h_list[-1], w)))
            else:
                h_list.append((h_ts.T[(layer/2) * self.hidden_size:(layer/2 + 1) * self.hidden_size]).T)

        gsn = GSN(
            inputs_hook        = (self.input_size, self.input),
            hiddens_hook       = (self.hidden_size, GSN.pack_hiddens(h_list)),
            params_hook        = self.gsn_params,
            outdir             = os.path.join(self.outdir, 'gsn_noisy/'),
            layers             = self.layers,
            walkbacks          = self.walkbacks,
            visible_activation = self.visible_activation_func,
            hidden_activation  = self.hidden_activation_func,
            input_sampling     = self.input_sampling,
            mrg                = self.mrg,
            tied_weights       = self.tied_weights,
            cost_function      = self.cost_function,
            cost_args          = self.cost_args,
            add_noise          = self.add_noise,
            noiseless_h1       = self.noiseless_h1,
            hidden_noise       = self.hidden_noise,
            hidden_noise_level = self.hidden_noise_level,
            input_noise        = self.input_noise,
            input_noise_level  = self.input_noise_level,
            noise_decay        = self.noise_decay,
            noise_annealing    = self.noise_annealing,
            image_width        = self.image_width,
            image_height       = self.image_height
        )

        cost = gsn.get_train_cost()
        monitors = gsn.get_monitors()  # frame-level error would be the 'mse' monitor from GSN
        x_sample_recon = gsn.get_outputs()

        # symbolic loop for sequence generation
        (x_ts, u_ts), updates_generate = theano.scan(lambda u_tm1: self.recurrent_step(None, u_tm1),
                                                     outputs_info=[None, self.generate_u0],
                                                     n_steps=self.n_steps,
                                                     name="rnngsn_generate_scan")

        return x_sample_recon, cost, monitors, updates_train, x_ts, updates_generate, u_ts[-1]

    def recurrent_step(self, x_t, u_tm1):
        """
        Performs one timestep for recurrence.

        Parameters
        ----------
        x_t : tensor
            The input at time t.
        u_tm1 : tensor
            The previous timestep (t-1) recurrent hiddens.

        Returns
        -------
        tuple
            Current generated visible x_t and recurrent u_t if generating (no x_t given as parameter),
            otherwise current recurrent u_t and hiddens h_t.
        """
        # If `x_t` is given, deterministic recurrence to compute the u_t. Otherwise, first generate.
        # Make current guess for hiddens based on U
        h_list = []
        for i in range(self.layers):
            if i % 2 == 0:
                log.debug("Using {0!s} and {1!s}".format(
                    self.recurrent_to_gsn_weights_list[(i+1) / 2], self.bias_list[i+1]))
                h = T.dot(u_tm1, self.recurrent_to_gsn_weights_list[(i+1) / 2]) + self.bias_list[i+1]
                h = self.hidden_activation_func(h)
                h_list.append(h)
        h_t = T.concatenate(h_list, axis=0)

        generate = x_t is None
        if generate:
            h_list_generate = [T.shape_padleft(h) for h in h_list]
            # create a GSN to generate x_t
            gsn = GSN(
                inputs_hook        = (self.input_size, self.input),
                hiddens_hook       = (self.hidden_size, T.concatenate(h_list_generate, axis=1)),
                params_hook        = self.gsn_params,
                outdir             = os.path.join(self.outdir, 'gsn_generate/'),
                layers             = self.layers,
                walkbacks          = self.walkbacks,
                visible_activation = self.visible_activation_func,
                hidden_activation  = self.hidden_activation_func,
                input_sampling     = self.input_sampling,
                mrg                = self.mrg,
                tied_weights       = self.tied_weights,
                cost_function      = self.cost_function,
                cost_args          = self.cost_args,
                add_noise          = self.add_noise,
                noiseless_h1       = self.noiseless_h1,
                hidden_noise       = self.hidden_noise,
                hidden_noise_level = self.hidden_noise_level,
                input_noise        = self.input_noise,
                input_noise_level  = self.input_noise_level,
                noise_decay        = self.noise_decay,
                noise_annealing    = self.noise_annealing,
                image_width        = self.image_width,
                image_height       = self.image_height
            )

            x_t = gsn.get_outputs().flatten()

        ua_t = T.dot(x_t, self.W_x_u) + T.dot(u_tm1, self.W_u_u) + self.recurrent_bias
        u_t = self.rnn_hidden_activation_func(ua_t)
        return [x_t, u_t] if generate else [u_t, h_t]

    def load_gsn_params(self, param_file):
        """
        Loads the parameters for the GSN only from param_file - used if the GSN was pre-trained

        Parameters
        ----------
        param_file : str
            Relative location of GSN parameters.

        Returns
        -------
        bool
            Whether or not successful.
        """
        param_file = os.path.realpath(param_file)

        # make sure it is a pickle file
        ftype = file_ops.get_file_type(param_file)
        if ftype == file_ops.PKL:
            log.debug("loading model %s parameters from %s",
                      str(type(self)), str(param_file))
            # try to grab the pickled params from the specified param_file path
            with open(param_file, 'r') as f:
                loaded_params = pickle.load(f)
            # set the GSN parameters
            for i, weight in enumerate(self.weights_list):
                weight.set_value(loaded_params[i])
            for i, bias in enumerate(self.bias_list):
                bias.set_value(loaded_params[i+len(self.weights_list)])
            return True
        # if get_file_type didn't return pkl or none, it wasn't a pickle file
        elif ftype:
            log.error("Param file %s doesn't have a supported pickle extension!", str(param_file))
            return False
        # if get_file_type returned none, it couldn't find the file
        else:
            log.error("Param file %s couldn't be found!", str(param_file))
            return False

    ####################
    # Model functions! #
    ####################
    def get_inputs(self):
        return self.input

    def get_outputs(self):
        return self.x_sample

    def generate(self, initial=None, n_steps=None):
        """
        Generate visible inputs from the model for `n_steps` and starting at recurrent hidden state `initial`.

        Parameters
        ----------
        initial : tensor
            Recurrent hidden state to start generation from.
        n_steps : int
            Number of generation steps to do.

        Returns
        -------
        tuple(array_like, array_like)
            The generated inputs and the ending recurrent hidden states.
        """
        # compile the generate function!
        if not hasattr(self, 'f_generate'):
            log.debug("compiling f_generate...")
            self.f_generate = function(inputs=[self.generate_u0, self.n_steps],
                                       outputs=[self.x_ts, self.u_t],
                                       updates=self.updates_generate)
            log.debug("compilation done!")

        initial = initial or self.u0.eval()
        n_steps = n_steps or self.generate_n_steps
        return self.f_generate(initial, n_steps)

    def get_train_cost(self):
        return self.cost

    def get_updates(self):
        return self.updates_train

    def get_monitors(self):
        return self.monitors

    def get_params(self):
        return self.params

    def save_args(self, args_file="rnngsn_config.pkl"):
        super(RNN_GSN, self).save_args(args_file)