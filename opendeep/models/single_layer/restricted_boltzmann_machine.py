"""
This module provides the RBM. http://deeplearning.net/tutorial/rbm.html

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.

Also see:
https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
for optimization tips and tricks.
"""

__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
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
from opendeep.utils.decorators import inherit_docs
from opendeep.models.model import Model
from opendeep.utils.nnet import get_weights, get_bias
from opendeep.utils.activation import get_activation_function, is_binary
from opendeep.utils.cost import binary_crossentropy

log = logging.getLogger(__name__)


@inherit_docs
class RBM(Model):
    """
    This is a probabilistic, energy-based model.
    Basic binary implementation from:
    http://deeplearning.net/tutorial/rnnrbm.html

    .. todo::
        Implement non-binary support for visible and hiddens (this means changing sampling method).

    """
    def __init__(self, inputs_hook=None, hiddens_hook=None, params_hook=None, outdir='outputs/rbm/',
                 input_size=None, hidden_size=None,
                 visible_activation='sigmoid', hidden_activation='sigmoid',
                 weights_init='uniform', weights_mean=0, weights_std=5e-3, weights_interval='montreal',
                 bias_init=0.0,
                 mrg=RNG_MRG.MRG_RandomStreams(1),
                 k=15):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        Parameters
        ----------
        inputs_hook : Tuple of (shape, variable)
            Routing information for the model to accept inputs from elsewhere. This is used for linking
            different models together. For now, it needs to include the shape information (normally the
            dimensionality of the input i.e. input_size).
        hiddens_hook : Tuple of (shape, variable)
            Routing information for the model to accept its hidden representation from elsewhere.
            This is used for linking different models together. For now, it needs to include the shape
            information (normally the dimensionality of the hiddens i.e. hidden_size).
        params_hook : List(theano shared variable)
            A list of model parameters (shared theano variables) that you should use when constructing
            this model (instead of initializing your own shared variables).
        outdir : str
            The directory you want outputs (parameters, images, etc.) to save to. If None, nothing will
            be saved.
        input_size : int
            The size (dimensionality) of the input to the RBM. If shape is provided in `inputs_hook`, this is optional.
            The :class:`Model` requires an `output_size`, which gets set to this value because the RBM is an
            unsupervised model. The output is a reconstruction of the input.
        hidden_size : int
            The size (dimensionality) of the hidden layer for the RBM.
        visible_activation : str or callable
            The nonlinear (or linear) visible activation to perform after the dot product from hiddens -> visible layer.
            This activation function should be appropriate for the input unit types, i.e. 'sigmoid' for binary inputs.
            See opendeep.utils.activation for a list of available activation functions. Alternatively, you can pass
            your own function to be used as long as it is callable.
        hidden_activation : str or callable
            The nonlinear (or linear) hidden activation to perform after the dot product from visible -> hiddens layer.
            See opendeep.utils.activation for a list of available activation functions. Alternatively, you can pass
            your own function to be used as long as it is callable.
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
        mrg : random
            A random number generator that is used when sampling. The RBM is a probabilistic model, so it relies a lot
            on sampling. I recommend using Theano's sandbox.rng_mrg.MRG_RandomStreams.
        k : int
            The k number of steps used for CD-k or PCD-k with Gibbs sampling. Basically, the number of samples
            generated from the model to train against reconstructing the original input.
        """
        # init Model to combine the defaults and config dictionaries with the initial parameters.
        super(RBM, self).__init__(**{arg: val for (arg, val) in locals().iteritems() if arg is not 'self'})

        ##################
        # specifications #
        ##################
        # grab info from the inputs_hook, hiddens_hook, or from parameters
        if inputs_hook is not None:  # inputs_hook is a tuple of (Shape, Input)
            assert len(inputs_hook) == 2, 'Expected inputs_hook to be tuple!'  # make sure inputs_hook is a tuple
            self.input = inputs_hook[1]
        else:
            # make the input a symbolic matrix
            self.input = T.matrix('V')

        # either grab the hidden's desired size from the parameter directly, or copy n_in
        hidden_size = hidden_size or self.input_size

        # get the number of steps k
        self.k = k

        # deal with hiddens_hook
        if hiddens_hook is not None:
            # make sure hiddens_hook is a tuple
            assert len(hiddens_hook) == 2, 'Expected hiddens_hook to be tuple!'
            hidden_size = hiddens_hook[0] or hidden_size
            self.hiddens_init = hiddens_hook[1]
        else:
            self.hiddens_init = None

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

        # make sure the sampling functions are appropriate for the activation functions.
        if is_binary(self.hidden_activation_func):
            self.hidden_sampling = mrg.binomial
        else:
            # TODO: implement non-binary activation
            log.error("Non-binary hidden activation not supported yet!")
            raise NotImplementedError("Non-binary hidden activation not supported yet!")

        ####################################################
        # parameters - make sure to deal with params_hook! #
        ####################################################
        if params_hook is not None:
            # make sure the params_hook has W (weights matrix) and bh, bv (bias vectors)
            assert len(params_hook) == 3, \
                "Expected 3 params (W, bv, bh) for RBM, found {0!s}!".format(len(params_hook))
            # doesn't matter if bv and bh are vectors or matrices.
            self.W, self.bv, self.bh = params_hook
            hidden_size = self.W.shape[1].eval()
        else:
            self.W = get_weights(weights_init=weights_init,
                                 shape=(self.input_size, hidden_size),
                                 name="W",
                                 rng=mrg,
                                 # if gaussian
                                 mean=weights_mean,
                                 std=weights_std,
                                 # if uniform
                                 interval=weights_interval)

            # grab the bias vectors
            self.bv = get_bias(shape=self.input_size, name="bv", init_values=bias_init)
            self.bh = get_bias(shape=hidden_size, name="bh", init_values=bias_init)

        # Finally have the parameters
        self.params = [self.W, self.bv, self.bh]

        # Create the RBM graph!
        self.cost, self.monitors, self.updates, self.v_sample, self.h_sample = self._build_rbm()

        log.debug("Initialized an RBM shape %s",
                  str((self.input_size, hidden_size)))

    def _build_rbm(self):
        """
        Creates the computation graph.

        Returns
        -------
        theano expression
            The cost expression - free energy.
        theano expression
            Monitor expression - binary cross-entropy to monitor training progress.
        dict
            Updates dictionary - updates from the Gibbs sampling process.
        tensor
            Last sample in the chain - last generated visible sample from the Gibbs process.
        tensor
            Last hidden sample in the chain from the Gibbs process.
        :rtype: List
        """
        # initialize from visibles if we aren't generating from some hiddens
        if self.hiddens_init is None:
            [_, v_chain, _, h_chain], updates = theano.scan(fn=lambda v: self._gibbs_step_vhv(v),
                                                            outputs_info=[None, self.input, None, None],
                                                            n_steps=self.k)
        # initialize from hiddens
        else:
            [_, v_chain, _, h_chain], updates = theano.scan(fn=lambda h: self._gibbs_step_hvh(h),
                                                            outputs_info=[None, None, None, self.hiddens_init],
                                                            n_steps=self.k)

        v_sample = v_chain[-1]
        h_sample = h_chain[-1]

        mean_v, _, _, _ = self._gibbs_step_vhv(v_sample)

        # some monitors
        # get rid of the -inf for the pseudo_log monitor (due to 0's and 1's in mean_v)
        # eps = 1e-8
        # zero_indices = T.eq(mean_v, 0.0).nonzero()
        # one_indices = T.eq(mean_v, 1.0).nonzero()
        # mean_v = T.inc_subtensor(x=mean_v[zero_indices], y=eps)
        # mean_v = T.inc_subtensor(x=mean_v[one_indices], y=-eps)
        pseudo_log = T.xlogx.xlogy0(self.input, mean_v) + T.xlogx.xlogy0(1 - self.input, 1 - mean_v)
        pseudo_log = pseudo_log.sum() / self.input.shape[0]
        crossentropy = T.mean(binary_crossentropy(mean_v, self.input))

        monitors = {'pseudo-log': pseudo_log, 'crossentropy': crossentropy}

        # the free-energy cost function!
        # consider v_sample constant when computing gradients on the cost function
        # this actually keeps v_sample from being considered in the gradient, to set gradient to 0 instead,
        # use theano.gradient.zero_grad
        v_sample_constant = theano.gradient.disconnected_grad(v_sample)
        # v_sample_constant = v_sample
        cost = (self.free_energy(self.input) - self.free_energy(v_sample_constant)) / self.input.shape[0]

        return cost, monitors, updates, v_sample, h_sample

    def _gibbs_step_vhv(self, v):
        """
        Single step in the Gibbs chain computing visible -> hidden -> visible.
        """
        # compute the hiddens and sample
        mean_h = self.hidden_activation_func(T.dot(v, self.W) + self.bh)
        h = self.hidden_sampling(size=mean_h.shape, n=1, p=mean_h,
                                 dtype=theano.config.floatX)
        # compute the visibles and sample
        mean_v = self.visible_activation_func(T.dot(h, self.W.T) + self.bv)
        v = self.visible_sampling(size=mean_v.shape, n=1, p=mean_v,
                                  dtype=theano.config.floatX)
        return mean_v, v, mean_h, h

    def _gibbs_step_hvh(self, h):
        """
        Single step in the Gibbs chain computing hidden -> visible -> hidden (for generative application).
        """
        # compute the visibles and sample
        mean_v = self.visible_activation_func(T.dot(h, self.W.T) + self.bv)
        v = self.visible_sampling(size=mean_v.shape, n=1, p=mean_v,
                                  dtype=theano.config.floatX)
        # compute the hiddens and sample
        mean_h = self.hidden_activation_func(T.dot(v, self.W) + self.bh)
        h = self.hidden_sampling(size=mean_h.shape, n=1, p=mean_h,
                                 dtype=theano.config.floatX)
        return mean_v, v, mean_h, h

    def free_energy(self, v):
        """
        The free-energy equation used for contrastive-divergence.

        Parameters
        ----------
        v : tensor
            The theano tensor representing the visible layer input.

        Returns
        -------
        theano expression
            The free energy calculation given the input tensor.
        """
        vbias_term  = -(v * self.bv).sum()
        hidden_term = -T.log(1 + T.exp(T.dot(v, self.W) + self.bh)).sum()
        return vbias_term + hidden_term

    ####################
    # Model functions! #
    ####################
    def get_inputs(self):
        return self.input

    def get_hiddens(self):
        return self.h_sample

    def get_outputs(self):
        return self.v_sample

    def generate(self, initial=None):
        log.exception("Generate not implemented yet for the RBM! Feel free to contribute :)")
        raise NotImplementedError("Generate not implemented yet for the RBM! Feel free to contribute :)")

    def get_train_cost(self):
        return self.cost

    def get_updates(self):
        return self.updates

    def get_monitors(self):
        return self.monitors

    def get_params(self):
        return self.params

    def save_args(self, args_file="rbm_config.pkl"):
        super(RBM, self).save_args(args_file)