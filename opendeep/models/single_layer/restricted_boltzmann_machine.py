"""
This module provides the RBM. http://deeplearning.net/tutorial/rbm.html

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.

Also see:
https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
for optimization tips and tricks.
"""
# standard libraries
import logging
# third party libraries
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG
# internal references
from opendeep.utils.decorators import inherit_docs
from opendeep.models.model import Model
from opendeep.utils.weights import get_weights, get_bias
from opendeep.utils.activation import get_activation_function, is_binary

log = logging.getLogger(__name__)


@inherit_docs
class RBM(Model):
    """
    This is a probabilistic, energy-based model.
    Basic binary implementation from:
    http://deeplearning.net/tutorial/rnnrbm.html
    and
    http://deeplearning.net/tutorial/rbm.html

    .. todo::
        Implement non-binary support for visible and hiddens (this means changing sampling method).

    """
    def __init__(self, inputs=None, hiddens=None, params=None, outdir='outputs/rbm/',
                 visible_activation='sigmoid', hidden_activation='sigmoid',
                 weights_init='uniform', weights_mean=0, weights_std=5e-3, weights_interval='glorot',
                 bias_init=0.0,
                 mrg=RNG_MRG.MRG_RandomStreams(1),
                 k=15):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

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
            The directory you want outputs (parameters, images, etc.) to save to. If None, nothing will
            be saved.
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
        initial_parameters = locals().copy()
        initial_parameters.pop('self')
        super(RBM, self).__init__(**initial_parameters)

        ##################
        # specifications #
        ##################
        if len(self.inputs) > 1:
            raise NotImplementedError("Expected 1 input to RBM, found %d. Please merge inputs before passing "
                                      "to the model!" % len(self.inputs))
        # self.inputs is a list of all the input expressions (we enforce only 1, so self.inputs[0] is the input)
        input_shape, self.input = self.inputs[0]
        if isinstance(input_shape, int):
            self.input_size = ((None,) * (self.input.ndim - 1)) + (input_shape,)
        else:
            self.input_size = input_shape
        assert self.input_size is not None, "Need to specify the shape for the last dimension of the input!"

        # our output space is the same as the input space
        self.output_size = self.input_size

        # grab hiddens
        # have only 1 hiddens
        assert len(self.hiddens) == 1, "Expected 1 `hiddens` param, found %d" % len(self.hiddens)
        self.hiddens = self.hiddens[0]
        if isinstance(self.hiddens, int):
            hidden_size = self.hiddens
            hiddens_init = None
        elif isinstance(self.hiddens, tuple):
            hidden_shape, hiddens_init = self.hiddens
            if isinstance(hidden_shape, int):
                hidden_size = hidden_shape
            else:
                hidden_size = hidden_shape[-1]
        else:
            raise AssertionError("Hiddens need to be an int or tuple of (shape, theano_expression), found %s" %
                                 type(self.hiddens))

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
        self.W = self.params.get(
            "W",
            get_weights(weights_init=weights_init,
                        shape=(self.input_size[-1], hidden_size),
                        name="W",
                        rng=mrg,
                        # if gaussian
                        mean=weights_mean,
                        std=weights_std,
                        # if uniform
                        interval=weights_interval)
        )
        self.b_v = self.params.get(
            "b_v",
            get_bias(shape=self.input_size[-1], name="b_v", init_values=bias_init)
        )
        self.b_h = self.params.get(
            "b_h",
            get_bias(shape=hidden_size, name="b_h", init_values=bias_init)
        )

        # Finally have the parameters
        self.params = {"W": self.W, "b_v": self.b_v, "b_h": self.b_h}

        ###############
        # computation #
        ###############
        # initialize from visibles if we aren't generating from some hiddens
        if hiddens_init is None:
            [_, v_chain, _, h_chain], self.updates = theano.scan(fn=self._gibbs_step_vhv,
                                                                 outputs_info=[None, self.input, None, None],
                                                                 n_steps=k)
        # initialize from hiddens
        else:
            [_, v_chain, _, h_chain], self.updates = theano.scan(fn=self._gibbs_step_hvh,
                                                                 outputs_info=[None, None, None, hiddens_init],
                                                                 n_steps=k)

        self.v_sample = v_chain[-1]
        self.h_sample = h_chain[-1]

        mean_v, _, _, _ = self._gibbs_step_vhv(self.v_sample)

        # the free-energy cost function!
        # consider v_sample constant when computing gradients on the cost function
        # this actually keeps v_sample from being considered in the gradient, to set gradient to 0 instead,
        # use theano.gradient.zero_grad
        v_sample_constant = theano.gradient.disconnected_grad(self.v_sample)
        # v_sample_constant = v_sample
        self.cost = (self.free_energy(self.input) - self.free_energy(v_sample_constant)) / self.input.shape[0]

        log.debug("Initialized an RBM shape %s",
                  str((self.input_size, hidden_size)))

    def _gibbs_step_vhv(self, v):
        """
        Single step in the Gibbs chain computing visible -> hidden -> visible.
        """
        # compute the hiddens and sample
        mean_h = self.hidden_activation_func(T.dot(v, self.W) + self.b_h)
        h = self.hidden_sampling(size=mean_h.shape, n=1, p=mean_h,
                                 dtype=theano.config.floatX)
        # compute the visibles and sample
        mean_v = self.visible_activation_func(T.dot(h, self.W.T) + self.b_v)
        v = self.visible_sampling(size=mean_v.shape, n=1, p=mean_v,
                                  dtype=theano.config.floatX)
        return mean_v, v, mean_h, h

    def _gibbs_step_hvh(self, h):
        """
        Single step in the Gibbs chain computing hidden -> visible -> hidden (for generative application).
        """
        # compute the visibles and sample
        mean_v = self.visible_activation_func(T.dot(h, self.W.T) + self.b_v)
        v = self.visible_sampling(size=mean_v.shape, n=1, p=mean_v,
                                  dtype=theano.config.floatX)
        # compute the hiddens and sample
        mean_h = self.hidden_activation_func(T.dot(v, self.W) + self.b_h)
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
        # vbias_term = -T.dot(v, self.b_v)
        # hidden_term = -T.sum(
        #     T.log(1 + T.exp(T.dot(v, self.W) + self.b_h)),
        #     axis=1
        # )
        vbias_term = -(v * self.b_v).sum()
        hidden_term = -T.log(1 + T.exp(T.dot(v, self.W) + self.b_h)).sum()
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

    def get_loss(self):
        return self.cost

    def get_updates(self):
        return self.updates

    def get_params(self):
        return self.params
