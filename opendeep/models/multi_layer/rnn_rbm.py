"""
.. module:: rnn_rbm

This module provides the RNN-RBM.
http://deeplearning.net/tutorial/rnnrbm.html
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
from opendeep.models.single_layer.restricted_boltzmann_machine import RBM
from opendeep.utils.nnet import get_weights, get_bias
from opendeep.utils.activation import get_activation_function, is_binary
from opendeep.utils import file_ops

log = logging.getLogger(__name__)

class RNN_RBM(Model):
    """
    http://deeplearning.net/tutorial/rnnrbm.html
    """
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
            # rbm parameters
            'hidden_size': None,
            'visible_activation': 'sigmoid',  # type of activation to use for visible activation
            'hidden_activation': 'sigmoid',  # type of activation to use for hidden activation
            'weights_init': 'uniform',  # either 'gaussian' or 'uniform' - how to initialize weights
            'weights_mean': 0,  # mean for gaussian weights init
            'weights_std': 0.005,  # standard deviation for gaussian weights init
            'weights_interval': 'montreal',  # if the weights_init was 'uniform', how to initialize from uniform
            'bias_init': 0.0,  # how to initialize the bias parameter
            'k': 15,  # the k steps used for CD-k or PCD-k with Gibbs sampling
            # general parameters
            'input_size': None,
            'mrg': RNG_MRG.MRG_RandomStreams(1),  # default random number generator from Theano
            'rng': numpy.random.RandomState(1),  #default random number generator from Numpy
            'outdir': 'outputs/rnnrbm/',  # the output directory for this model's outputs
    }

    def __init__(self, inputs_hook=None, hiddens_hook=None, params_hook=None, config=None, defaults=default,
                 input_size=None, hidden_size=None, visible_activation=None, hidden_activation=None,
                 weights_init=None, weights_mean=None, weights_std=None, weights_interval=None, bias_init=None,
                 mrg=None, rng=None, k=None, outdir=None, rnn_hidden_size=None, rnn_hidden_activation=None,
                 rnn_weights_init=None, rnn_weights_mean=None, rnn_weights_std=None,
                 rnn_weights_interval=None, rnn_bias_init=None, generate_n_steps=None):
        # init Model to combine the defaults and config dictionaries with the initial parameters.
        super(RNN_RBM, self).__init__(**{arg: val for (arg, val) in locals().iteritems() if arg is not 'self'})
        # all configuration parameters are now in self!

        ##################
        # specifications #
        ##################
        # grab info from the inputs_hook, hiddens_hook, or from parameters
        if self.inputs_hook is not None:  # inputs_hook is a tuple of (Shape, Input)
            raise NotImplementedError("Inputs_hook not implemented yet for RNN_RBM")
        else:
            # make the input a symbolic matrix - a sequence of inputs
            self.input = T.fmatrix('Vs')

        # set an initial value for the recurrent hiddens
        self.u0 = T.zeros((self.rnn_hidden_size,))

        # make a symbolic vector for the initial recurrent hiddens value to use during generation for the model
        self.generate_u0 = T.fvector("generate_u0")

        # either grab the hidden's desired size from the parameter directly, or copy n_in
        self.hidden_size = self.hidden_size or self.input_size

        # deal with hiddens_hook
        if self.hiddens_hook is not None:
            raise NotImplementedError("Hiddnes_hook not implemented yet for RNN_RBM")


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

        # make sure the sampling functions are appropriate for the activation functions.
        if is_binary(self.hidden_activation_func):
            self.hidden_sampling = self.mrg.binomial
        else:
            # TODO: implement non-binary activation
            log.error("Non-binary hidden activation not supported yet!")
            raise NotImplementedError("Non-binary hidden activation not supported yet!")

        # recurrent hidden activation function!
        self.rnn_hidden_activation_func = get_activation_function(self.rnn_hidden_activation)

        # symbolic scalar for how many recurrent steps to use during generation from the model
        self.n_steps = T.iscalar("generate_n_steps")

        ####################################################
        # parameters - make sure to deal with params_hook! #
        ####################################################
        if self.params_hook is not None:
            # make sure the params_hook has W (weights matrix) and bh, bv (bias vectors)
            assert len(self.params_hook) == 8, \
                "Expected 8 params (W, bv, bh, Wuh, Wuv, Wvu, Wuu, bu) for RBM, found {0!s}!".format(
                    len(self.params_hook)
                )
            self.W, self.bv, self.bh, self.Wuh, self.Wuv, self.Wvu, self.Wuu, self.bu = self.params_hook
        else:
            # RBM weight params
            self.W = get_weights(weights_init=self.weights_init,
                                 shape=(self.input_size, self.hidden_size),
                                 name="W",
                                 rng=self.rng,
                                 # if gaussian
                                 mean=self.weights_mean,
                                 std=self.weights_std,
                                 # if uniform
                                 interval=self.weights_interval)
            # RNN weight params
            self.Wuh = get_weights(weights_init=self.rnn_weights_init,
                                   shape=(self.rnn_hidden_size, self.hidden_size),
                                   name="Wuh",
                                   rng=self.rng,
                                   # if gaussian
                                   mean=self.rnn_weights_mean,
                                   std=self.rnn_weights_std,
                                   # if uniform
                                   interval=self.rnn_weights_interval)

            self.Wuv = get_weights(weights_init=self.rnn_weights_init,
                                   shape=(self.rnn_hidden_size, self.input_size),
                                   name="Wuv",
                                   rng=self.rng,
                                   # if gaussian
                                   mean=self.rnn_weights_mean,
                                   std=self.rnn_weights_std,
                                   # if uniform
                                   interval=self.rnn_weights_interval)

            self.Wvu = get_weights(weights_init=self.rnn_weights_init,
                                   shape=(self.input_size, self.rnn_hidden_size),
                                   name="Wvu",
                                   rng=self.rng,
                                   # if gaussian
                                   mean=self.rnn_weights_mean,
                                   std=self.rnn_weights_std,
                                   # if uniform
                                   interval=self.rnn_weights_interval)

            self.Wuu = get_weights(weights_init=self.rnn_weights_init,
                                   shape=(self.rnn_hidden_size, self.rnn_hidden_size),
                                   name="Wuu",
                                   rng=self.rng,
                                   # if gaussian
                                   mean=self.rnn_weights_mean,
                                   std=self.rnn_weights_std,
                                   # if uniform
                                   interval=self.rnn_weights_interval)

            # grab the bias vectors
            # rbm biases
            self.bv = get_bias(shape=self.input_size, name="bv", init_values=self.bias_init)
            self.bh = get_bias(shape=self.hidden_size, name="bh", init_values=self.bias_init)
            # rnn bias
            self.bu = get_bias(shape=self.rnn_hidden_size, name="bu", init_values=self.rnn_bias_init)

        # Finally have the parameters
        self.params = [self.W, self.bv, self.bh, self.Wuh, self.Wuv, self.Wvu, self.Wuu, self.bu]

        # Create the RNN-RBM graph!
        self.v_sample, self.cost, self.monitors, self.updates_train, self.v_ts, self.updates_generate, self.u_t = \
            self.build_rnnrbm()

        log.info("Initialized an RNN-RBM!")

    def build_rnnrbm(self):
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
        # {bv_t, bh_t, 1 <= t <= T} given v. Conditional RBMs can then be trained
        # in batches using those parameters.
        (u_ts, bv_ts, bh_ts), updates_train = theano.scan(fn=lambda v_t, u_tm1: self.recurrence(v_t, u_tm1),
                                                       sequences=self.input,
                                                       outputs_info=[self.u0, None, None],
                                                       name="rnnrbm_computation_scan")

        rbm = RBM(inputs_hook=(self.input_size, self.input),
                  params_hook=(self.W, bv_ts[:], bh_ts[:]),
                  k=self.k,
                  outdir=self.outdir,
                  mrg=self.mrg,
                  rng=self.rng)
        v_sample    = rbm.get_outputs()
        cost        = rbm.get_train_cost()
        monitors    = rbm.get_monitors()
        updates_rbm = rbm.get_updates()

        # make another chain to determine frame-level accuracy/error
        rbm = RBM(inputs_hook=(self.input_size, self.input[:-1]),
                  params_hook=(self.W, bv_ts[1:], bh_ts[1:]),
                  k=self.k,
                  outdir=self.outdir,
                  mrg=self.mrg,
                  rng=self.rng)

        v_prediction    = rbm.get_outputs()
        updates_predict = rbm.get_updates()

        frame_level_mse = T.mean(T.sqr(v_sample[1:] - v_prediction), axis=0)
        frame_level_error = T.mean(frame_level_mse)
        # add the frame-level error to the monitors
        monitors['frame_error'] = frame_level_error

        updates_train.update(updates_rbm)
        updates_train.update(updates_predict)

        # symbolic loop for sequence generation
        (v_ts, u_ts), updates_generate = theano.scan(lambda u_tm1, *_: self.recurrence(None, u_tm1),
                                                     outputs_info=[None, self.generate_u0],
                                                     non_sequences=self.params,
                                                     n_steps=self.n_steps,
                                                     name="rnnrbm_generate_scan")

        return v_sample, cost, monitors, updates_train, v_ts, updates_generate, u_ts[-1]

    def recurrence(self, v_t, u_tm1):
        """
        The single recurrent step for the model

        :param v_t: the visible layer at time t
        :type v_t: tensor

        :param u_tm1: the recurrent hidden layer at time t-1
        :type u_tm1: tensor

        :return: tuple of current v_t and updates if generating from model, otherwise, current u_t and rbm bias params
        :rtype: tuple
        """
        # generate the current rbm bias params
        bv_t = self.bv + T.dot(u_tm1, self.Wuv)
        bh_t = self.bh + T.dot(u_tm1, self.Wuh)
        # if we should be generating from the recurrent model
        generate = v_t is None
        updates = None
        if generate:
            rbm = RBM(inputs_hook=(self.input_size, T.zeros((self.input_size,))),
                      params_hook=(self.W, bv_t, bh_t),
                      k=self.k,
                      outdir=self.outdir,
                      mrg=self.mrg,
                      rng=self.rng)
            v_t = rbm.get_outputs()
            updates = rbm.get_updates()

        # update recurrent hiddens
        u_t = T.tanh(self.bu + T.dot(v_t, self.Wvu) + T.dot(u_tm1, self.Wuu))
        return ([v_t, u_t], updates) if generate else (u_t, bv_t, bh_t)

    def load_rbm_params(self, param_file):
        """
        Loads the parameters for the RBM only from param_file - used if the RBM was pre-trained

        :param param_file: location of rbm parameters
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
            #############################################################################
            # set the W, bv, and bh values (make sure same order as saved in RBM class) #
            #############################################################################
            # TODO: Switch back to correct order after testing mnist rnn-rbm
            self.W.set_value(loaded_params[0])
            self.bv.set_value(loaded_params[2])
            self.bh.set_value(loaded_params[1])
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
        return self.v_sample

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
            self.f_generate = function(inputs=[self. generate_u0, self.n_steps],
                                       outputs=[self.v_ts, self.u_t],
                                       updates=self.updates_generate)

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

    def save_args(self, args_file="rnnrbm_config.pkl"):
        super(RNN_RBM, self).save_args(args_file)