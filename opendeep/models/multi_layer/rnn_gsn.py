"""
.. module:: rnn_gsn

This module provides the RNN-GSN.
TODO: <link to my paper when it is on arxiv>
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
import numpy
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG
# internal references
from opendeep import function
from opendeep.models.model import Model
from opendeep.models.multi_layer.generative_stochastic_network import GSN
from opendeep.utils.nnet import get_weights, get_bias
from opendeep.utils.activation import get_activation_function, is_binary
from opendeep.utils import file_ops
from opendeep.utils.cost import get_cost_function

log = logging.getLogger(__name__)

class RNN_GSN(Model):
    default = {
            # recurrent parameters
            'rnn_hidden_size': None,
            'rnn_hidden_activation': 'relu',  # type of activation to use for recurrent hidden activation
            'rnn_weights_init': 'identity',  # how to initialize weights
            'rnn_weights_mean': 0,  # mean for gaussian weights init
            'rnn_weights_std': 0.005,  # standard deviation for gaussian weights init
            'rnn_weights_interval': 'montreal',  # how to initialize from uniform
            'rnn_bias_init': 0.0,  # how to initialize the bias parameter
            'generate_n_steps': 200,  # how many steps to generate
            # gsn parameters
            "layers": 3,  # number of hidden layers to use
            "walkbacks": 5,  # number of walkbacks (generally 2*layers) - need enough to propagate to visible layer
            "input_size": None,  # number of input features - please specify for your dataset!
            "hidden_size": 1500,  # number of hidden units in each layer
            "visible_activation": 'sigmoid',  # activation for visible layer - make appropriate for input data type.
            "hidden_activation": 'tanh',  # activation for hidden layers
            "input_sampling": True,  # whether to sample at each walkback step - makes it like Gibbs sampling.
            "mrg": RNG_MRG.MRG_RandomStreams(1),  # default random number generator from Theano
            "tied_weights": True,  # whether to tie the weights between layers (use transpose from higher to lower)
            "weights_init": "uniform",  # how to initialize weights
            'weights_interval': 'montreal',  # if the weights_init was 'uniform', how to initialize from uniform
            'weights_mean': 0,  # mean for gaussian weights init
            'weights_std': 0.005,  # standard deviation for gaussian weights init
            'bias_init': 0.0,  # how to initialize the bias parameter
            # train param
            "cost_function": 'binary_crossentropy',  # the cost function for training; make appropriate for input type.
            # noise parameters
            "noise_decay": 'exponential',  # noise schedule algorithm
            "noise_annealing": 1.0,  # no noise schedule by default
            "add_noise": True,  # whether to add noise throughout the network's hidden layers
            "noiseless_h1": True,  # whether to keep the first hidden layer uncorrupted
            "hidden_add_noise_sigma": 2,  # sigma value for adding the gaussian hidden layer noise
            "input_salt_and_pepper": 0.4,  # the salt and pepper value for inputs corruption
            # data parameters
            "outdir": 'outputs/rnngsn/',  # base directory to output various files
            "image_width": None,  # if the input is an image, its width
            "image_height": None,  # if the input is an image, its height
            "vis_init": False
    }

    def __init__(self, inputs_hook=None, hiddens_hook=None, params_hook=None,
                 config=None, defaults=default,
                 input_size=None, hidden_size=None,
                 layers=None, walkbacks=None,
                 visible_activation=None, hidden_activation=None,
                 input_sampling=None, mrg=None,
                 tied_weights=None, weights_init=None, weights_interval=None, weights_mean=None, weights_std=None,
                 bias_init=None,
                 cost_function=None,
                 noise_decay=None, noise_annealing=None,
                 add_noise=None, noiseless_h1=None, hidden_add_noise_sigma=None, input_salt_and_pepper=None,
                 outdir=None,
                 image_width=None, image_height=None,
                 vis_init=None,
                 rnn_hidden_size=None, rnn_hidden_activation=None,
                 rnn_weights_init=None, rnn_weights_mean=None, rnn_weights_std=None, rnn_weights_interval=None,
                 rnn_bias_init=None, generate_n_steps=None):
        # init Model to combine the defaults and config dictionaries with the initial parameters.
        initial_parameters = locals().copy()
        initial_parameters.pop('self')
        super(RNN_GSN, self).__init__(**initial_parameters)
        # all configuration parameters are now in self!

        ##################
        # specifications #
        ##################
        # grab info from the inputs_hook, hiddens_hook, or from parameters
        if self.inputs_hook is not None:  # inputs_hook is a tuple of (Shape, Input)
            raise NotImplementedError("Inputs_hook not implemented yet for RNN-GSN")
        else:
            # make the input a symbolic matrix - a sequence of inputs
            self.input = T.matrix('Xs')

        # set an initial value for the recurrent hiddens
        self.u0 = T.zeros((self.rnn_hidden_size,))

        # make a symbolic vector for the initial recurrent hiddens value to use during generation for the model
        self.generate_u0 = T.vector("generate_u0")

        # either grab the hidden's desired size from the parameter directly, or copy n_in
        self.hidden_size = self.hidden_size or self.input_size

        # deal with hiddens_hook
        if self.hiddens_hook is not None:
            raise NotImplementedError("Hiddens_hook not implemented yet for RNN-GSN")

        # other specifications
        # visible activation function!
        self.visible_activation_func = get_activation_function(self.visible_activation)

        # make sure the sampling functions are appropriate for the activation functions.
        if is_binary(self.visible_activation_func):
            self.visible_sampling = self.mrg.binomial
        else:
            # TODO: implement non-binary activation
            log.error("Non-binary visible activation not supported yet!")
            raise NotImplementedError("Non-binary visible activation not supported yet!")

        # hidden activation function!
        self.hidden_activation_func = get_activation_function(self.hidden_activation)

        # recurrent hidden activation function!
        self.rnn_hidden_activation_func = get_activation_function(self.rnn_hidden_activation)

        # Cost function
        self.cost_function = get_cost_function(self.cost_function)

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
            self.weights_list = [get_weights(weights_init=self.weights_init,
                                             shape=(self.layer_sizes[i], self.layer_sizes[i + 1]),
                                             name="W_{0!s}_{1!s}".format(i, i + 1),
                                             # if gaussian
                                             mean=self.weights_mean,
                                             std=self.weights_std,
                                             # if uniform
                                             interval=self.weights_interval)
                                 for i in range(self.layers)]
            # add more weights if we aren't tying weights between layers (need to add for higher-lower layers now)
            if not self.tied_weights:
                self.weights_list.extend(
                    [get_weights(weights_init=self.weights_init,
                                 shape=(self.layer_sizes[i + 1], self.layer_sizes[i]),
                                 name="W_{0!s}_{1!s}".format(i + 1, i),
                                 # if gaussian
                                 mean=self.weights_mean,
                                 std=self.weights_std,
                                 # if uniform
                                 interval=self.weights_interval)
                     for i in reversed(range(self.layers))]
                )
            # initialize each layer bias to 0's.
            self.bias_list = [get_bias(shape=(self.layer_sizes[i],),
                                       name='b_' + str(i),
                                       init_values=self.bias_init)
                              for i in range(self.layers + 1)]

            self.recurrent_to_gsn_weights_list = [
                get_weights(weights_init=self.rnn_weights_init,
                            shape=(self.rnn_hidden_size, self.layer_sizes[layer]),
                            name="W_u_h{0!s}".format(layer),
                            # if gaussian
                            mean=self.rnn_weights_mean,
                            std=self.rnn_weights_std,
                            # if uniform
                            interval=self.rnn_weights_interval)
                for layer in range(self.layers + 1) if layer % 2 != 0
            ]
            self.W_u_u = get_weights(weights_init=self.rnn_weights_init,
                                     shape=(self.rnn_hidden_size, self.rnn_hidden_size),
                                     name="W_u_u",
                                     # if gaussian
                                     mean=self.rnn_weights_mean,
                                     std=self.rnn_weights_std,
                                     #if uniform
                                     interval=self.rnn_weights_interval)
            self.W_x_u = get_weights(weights_init=self.rnn_weights_init,
                                     shape=(self.input_size, self.rnn_hidden_size),
                                     name="W_x_u",
                                     # if gaussian
                                     mean=self.rnn_weights_mean,
                                     std=self.rnn_weights_std,
                                     # if uniform
                                     interval=self.rnn_weights_interval)
            self.recurrent_bias = get_bias(shape=(self.rnn_hidden_size,),
                                           name="b_u",
                                           init_values=self.rnn_bias_init)

        # build the params of the model into a list
        self.gsn_params = self.weights_list + self.bias_list
        self.params = self.gsn_params + \
                      self.recurrent_to_gsn_weights_list + \
                      [self.W_u_u, self.W_x_u, self.recurrent_bias]
        log.debug("rnn-gsn params: %s", str(self.params))

        # Create the RNN-GSN graph!
        self.x_sample, self.cost, self.monitors, self.updates_train, self.x_ts, self.updates_generate, self.u_t = \
            self.build_rnngsn()

        log.info("Initialized an RNN-GSN!")

    def build_rnngsn(self):
        """
        Creates the updates and other return variables for the computation graph

        :return: the sample at the end of the computation graph,
        the train cost function,
        the train monitors,
        the computation updates,
        the generated visible list,
        the generated computation updates
        :rtype: List
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

        gsn = GSN(inputs_hook           = (self.input_size, self.input),
                  hiddens_hook          = (self.hidden_size, GSN.pack_hiddens(h_list)),
                  params_hook           = self.weights_list+self.bias_list,
                  layers                = self.layers,
                  walkbacks             = self.walkbacks,
                  visible_activation    = self.visible_activation_func,
                  hidden_activation     = self.hidden_activation_func,
                  input_sampling        = self.input_sampling,
                  mrg                   = self.mrg,
                  tied_weights          = self.tied_weights,
                  cost_function         = self.cost_function,
                  noise_decay           = self.noise_decay,
                  noise_annealing       = self.noise_annealing,
                  add_noise             = self.add_noise,
                  noiseless_h1          = self.noiseless_h1,
                  hidden_add_noise_sigma= self.hidden_add_noise_sigma,
                  input_salt_and_pepper = self.input_salt_and_pepper,
                  outdir                = os.path.join(self.outdir, 'gsn_noisy/'),
                  image_width           = self.image_width,
                  image_height          = self.image_height,
                  vis_init              = self.vis_init)

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
        performs one timestep for recurrence
        :param x_t: the input at time t
        :type x_t: theano tensor
        :param u_tm1: the previous time recurrent hiddens
        :type u_tm1: theano tensor
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
            gsn = GSN(inputs_hook=(self.input_size, self.input),
                      hiddens_hook=(self.hidden_size, T.concatenate(h_list_generate, axis=1)),
                      params_hook=self.weights_list + self.bias_list,
                      layers=self.layers,
                      walkbacks=self.walkbacks,
                      visible_activation=self.visible_activation_func,
                      hidden_activation=self.hidden_activation_func,
                      input_sampling=self.input_sampling,
                      mrg=self.mrg,
                      tied_weights=self.tied_weights,
                      cost_function=self.cost_function,
                      noise_decay=self.noise_decay,
                      noise_annealing=self.noise_annealing,
                      add_noise=self.add_noise,
                      noiseless_h1=self.noiseless_h1,
                      hidden_add_noise_sigma=self.hidden_add_noise_sigma,
                      input_salt_and_pepper=self.input_salt_and_pepper,
                      outdir=os.path.join(self.outdir, 'gsn_generate/'),
                      image_width=self.image_width,
                      image_height=self.image_height,
                      vis_init=self.vis_init)
            x_t = gsn.get_outputs().flatten()

        ua_t = T.dot(x_t, self.W_x_u) + T.dot(u_tm1, self.W_u_u) + self.recurrent_bias
        u_t = self.rnn_hidden_activation_func(ua_t)
        return [x_t, u_t] if generate else [u_t, h_t]

    def load_gsn_params(self, param_file):
        """
        Loads the parameters for the GSN only from param_file - used if the GSN was pre-trained

        :param param_file: location of GSN parameters
        :type param_file: string

        :return: whether successful
        :rtype: boolean
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
        Generate visible inputs from the model for n_steps and starting at recurrent hidden state initial

        :param initial: recurrent hidden state to start generation from
        :type initial: tensor

        :param n_steps: number of generation steps to do
        :type n_steps: int

        :return: the generated inputs and the ending recurrent hidden state
        :rtype: matrix, matrix
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