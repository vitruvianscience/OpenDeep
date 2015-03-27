"""
.. module:: basic

This module provides the most basic neural net layers.
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
import theano.tensor as T
# internal references
from opendeep.models.model import Model
from opendeep.utils.nnet import get_weights_gaussian, get_weights_uniform, get_bias
from opendeep.utils.activation import get_activation_function
from opendeep.utils.cost import get_cost_function

log = logging.getLogger(__name__)

class BasicLayer(Model):
    """
    This is your basic input -> nonlinear(output) layer. No hidden representation.
    """
    default = {
        'activation': 'rectifier',  # type of activation function to use for output
        'cost': 'mse',  # the cost function to use for supervised training - comparing outputs to target labels.
        'cost_args': {},  # extra arguments to give the cost function (such as std's for gaussian LL). normally not used
        'weights_init': 'gaussian',  # either 'gaussian' or 'uniform' - how to initialize weights
        'weights_mean': 0,  # mean for gaussian weights init
        'weights_std': 0.005,  # standard deviation for gaussian weights init
        'weights_interval': 'montreal',  # if the weights_init was 'uniform', how to initialize from uniform
        'bias_init': 0.0,  # how to initialize the bias parameter
        'input_size': None,
        'output_size': None
    }
    def __init__(self, inputs_hook=None, config=None, defaults=default, params_hook=None,
                 input_size=None, output_size=None, activation=None, cost=None, cost_args=None,
                 weights_init=None, weights_mean=None, weights_std=None, weights_interval=None, bias_init=None,
                 **kwargs):
        # init Model to combine the defaults and config dictionaries with the initial parameters.
        super(BasicLayer, self).__init__(**{arg: val for (arg, val) in locals().iteritems() if arg is not 'self'})
        # all configuration parameters are now in self!

        ##################
        # specifications #
        ##################
        # grab info from the inputs_hook, or from parameters
        if self.inputs_hook is not None:  # inputs_hook is a tuple of (Shape, Input)
            assert len(self.inputs_hook) == 2, 'Expected inputs_hook to be tuple!'  # make sure inputs_hook is a tuple
            self.input_size = self.inputs_hook[0] or self.input_size
            self.input = self.inputs_hook[1]
        else:
            # make the input a symbolic matrix
            self.input = T.fmatrix('X')

        # now that we have the input specs, define the output 'target' variable to be used in supervised training!
        self.target = T.fmatrix('Y')

        # either grab the output's desired size from the parameter directly, or copy n_in
        self.output_size = self.output_size or self.input_size

        # other specifications
        # activation function!
        # if a string name was given, look up the correct function from our utils.
        if isinstance(self.activation, basestring):
            activation_func = get_activation_function(self.activation)
        # otherwise, if a 'callable' was passed (i.e. custom function), use that directly.
        else:
            assert callable(self.activation), "Activation function either needs to be a string name or callable!"
            activation_func = self.activation

        # cost function!
        # if a string name was given, look up the correct function from our utils.
        if isinstance(self.cost, basestring):
            cost_func = get_cost_function(self.cost)
        # otherwise, if a 'callable' was passed (i.e. custom function), use that directly.
        else:
            assert callable(self.cost), "Cost function either needs to be a string name or callable!"
            cost_func = self.cost

        ####################################################
        # parameters - make sure to deal with params_hook! #
        ####################################################
        if self.params_hook is not None:
            # make sure the params_hook has W (weights matrix) and b (bias vector)
            assert len(self.params_hook) == 2, \
                "Expected 2 params (W and b) for BasicLayer, found {0!s}!".format(len(self.params_hook))
            W, b = self.params_hook
        else:
            # if we are initializing weights from a gaussian distribution
            if self.weights_init.lower() == 'gaussian':
                W = get_weights_gaussian(
                    shape=(self.input_size, self.output_size), mean=self.weights_mean, std=self.weights_std, name="W"
                )
            # if we are initializing weights from a uniform distribution
            elif self.weights_init.lower() == 'uniform':
                interval = self.weights_interval
                W = get_weights_uniform(shape=(self.input_size, self.output_size), interval=interval, name="W")
            # otherwise not implemented
            else:
                log.error("Did not recognize weights_init %s! Pleas try gaussian or uniform" %
                          str(self.weights_init))
                raise NotImplementedError("Did not recognize weights_init %s! Pleas try gaussian or uniform" %
                                          str(self.weights_init))

            # grab the bias vector
            b = get_bias(shape=self.output_size, name="b", init_values=self.bias_init)

        # Finally have the two parameters - weights matrix W and bias vector b. That is all!
        self.params = [W, b]

        ###############
        # computation #
        ###############
        # Here is the meat of the computation transforming input -> output
        # It simply involves a matrix multiplication of inputs*weights, adding the bias vector, and then passing
        # the result through our activation function (normally something nonlinear such as: max(0, output))
        self.output = activation_func(T.dot(self.input, W) + b)

        # now to define the cost of the model - use the cost function to compare our output with the target value.
        self.cost = cost_func(output=self.output, target=self.target, **self.cost_args)

        log.debug("Initialized a basic fully-connected layer with shape %s and activation: %s",
                  str((self.input_size, self.output_size)), str(self.activation))

    def get_inputs(self):
        return [self.input]

    def get_outputs(self):
        return self.output

    def get_targets(self):
        return [self.target]

    def get_train_cost(self):
        return self.cost

    def get_params(self):
        return self.params

    def save_args(self, args_file="basiclayer_config.pkl"):
        super(BasicLayer, self).save_args(args_file)


class SoftmaxLayer(BasicLayer):
    """
    The softmax layer is meant as a last-step prediction layer using the softmax activation function -
    this class exists to provide easy access to methods for errors and log-likelihood for a given truth label y.

    It is a special subclass of the FullyConnectedLayer, with the activation function forced to be 'softmax'
    """
    default = {'cost': 'nll',  # the cost function to use
               'out_as_probs': False  # whether output is class guess (False) or vector of class probabilities (True)
               }
    def __init__(self, inputs_hook=None, config=None, defaults=default, params_hook=None,
                 input_size=None, output_size=None, weights_init=None, weights_mean=None, weights_std=None,
                 weights_interval=None, bias_init=None, cost=None, cost_args=None, activation='softmax',
                 out_as_probs=None):
        # grab what cost to use
        if cost is None:
            if config is not None:
                cost = config.get('cost', defaults.get('cost'))
            else:
                cost = defaults.get('cost')
        # see if we want to output a class guess or vector of probabilities
        if out_as_probs is None:
            if config is not None:
                out_as_probs = config.get('out_as_probs', defaults.get('out_as_probs'))
            else:
                out_as_probs = defaults.get('out_as_probs')

        # init the fully connected generic layer with a softmax activation function
        super(SoftmaxLayer, self).__init__(inputs_hook=inputs_hook,
                                           params_hook=params_hook,
                                           activation='softmax',
                                           cost=cost,
                                           cost_args=cost_args,
                                           config=config,
                                           input_size=input_size,
                                           output_size=output_size,
                                           weights_init=weights_init,
                                           weights_mean=weights_mean,
                                           weights_std=weights_std,
                                           weights_interval=weights_interval,
                                           bias_init=bias_init,
                                           out_as_probs=out_as_probs)
        # target_flag shows whether or not we are using the super class's targets, or making our own
        # integer vector for targets. This becomes true if we are using the nll cost, since it requires
        # integer labels.
        self.target_flag = False

        # the outputs of the layer are the probabilities of being in a given class
        self.p_y_given_x = super(SoftmaxLayer, self).get_outputs()
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.y = T.concatenate(self.get_targets())

        # if cost was nll, set self.cost to negative log likelihood
        # this is what gets returned as the train cost for the BasicLayer superclass.
        if cost.lower() == 'nll':
            log.debug('Using softmax negative log-likelihood cost!!')
            # nll requires integer targets 'y'.
            self.target_flag = True
            self.y = T.fvector('y')
            self.cost = self.negative_log_likelihood()

    def get_outputs(self):
        # if we aren't asking for the class probabilities, return the argmax (gives the index of highest probability)
        if not self.out_as_probs:
            return self.get_argmax_prediction()
        # otherwise, give the output as normal, which is the vector of probabilities
        else:
            return self.p_y_given_x

    def get_targets(self):
        """
        returns the target 'labels' to compare against

        :return: symbolic tensor
        """
        # return our integer targets, or default to superclass
        if self.target_flag:
            return [self.y]
        else:
            return super(SoftmaxLayer, self).get_targets()

    def get_monitors(self):
        # grab the basiclayer's monitors
        monitors = super(SoftmaxLayer, self).get_monitors()
        # if this softmax layer is using integer classes, add the 'error' monitor.
        if self.target_flag:
            monitors.update({'softmax_error': self.errors()})
        return monitors

    def negative_log_likelihood(self):
        """Return the mean of the negative log-likelihood of the prediction
            of this model under a given target distribution.

            .. math::

                \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
                \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                    \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

            Note: we use the mean instead of the sum so that
                  the learning rate is less dependent on the batch size
            """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(self.y.shape[0]), T.cast(self.y, 'int32')])

    def errors(self):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero-one
        loss over the size of the minibatch
        """

        # check if y has same dimension of y_pred
        if self.y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', self.y.type, 'y_pred', self.y_pred.type)
            )
        # the T.neq operator returns a vector of 0s and 1s, where 1
        # represents a mistake in prediction
        return T.mean(T.neq(self.y_pred, self.y))

    def get_argmax_prediction(self):
        # return the argmax y_pred class
        return self.y_pred

    def save_args(self, args_file="softmax_config.pkl"):
        super(SoftmaxLayer, self).save_args(args_file)