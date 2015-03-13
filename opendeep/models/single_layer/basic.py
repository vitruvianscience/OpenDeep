"""
.. module:: basic

This module provides the most basic neural net layers.
"""

__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import logging
# third party libraries
import theano.tensor as T
# internal references
from opendeep.models.model import Model
from opendeep.utils.nnet import get_weights_gaussian, get_weights_uniform, get_bias
from opendeep.utils.activation import get_activation_function

log = logging.getLogger(__name__)

class BasicLayer(Model):
    """
    This is your basic input -> nonlinear(output) layer. No hidden representation.
    """
    default = {
        'activation': 'rectifier',  # type of activation function to use for output
        'weights_init': 'gaussian',  # either 'gaussian' or 'uniform' - how to initialize weights
        'weights_mean': 0,  # mean for gaussian weights init
        'weights_std': 0.005,  # standard deviation for gaussian weights init
        'weights_interval': 'montreal',  # if the weights_init was 'uniform', how to initialize from uniform
        'bias_init': 0.0,  # how to initialize the bias parameter
    }
    def __init__(self, inputs_hook=None, config=None, defaults=default, params_hook=None, input_size=None, output_size=None, activation=None,
                 weights_init=None, weights_mean=None, weights_std=None, weights_interval=None, bias_init=None):
        # init Model to combine the defaults and config dictionaries.
        super(BasicLayer, self).__init__(config, defaults)
        # all configuration parameters are now in self.args

        ##################
        # specifications #
        ##################
        # grab info from the inputs_hook, or from parameters
        if inputs_hook:  # inputs_hook is a tuple of (Shape, Input)
            assert len(inputs_hook) == 2  # make sure inputs_hook is a tuple
            input_size = inputs_hook[0] or input_size
            self.input = inputs_hook[1]
        else:
            input_size = input_size or self.args.get('input_size')  # either grab from the parameter directly or self.args config
            self.input = T.fmatrix('X')  # make the input a symbolic matrix
        output_size = output_size or self.args.get('output_size') or input_size  # either grab from the parameter directly, self.args config, or copy n_in

        # other specifications
        weights_init = weights_init or self.args.get('weights_init')
        # for gaussian weights
        mean         = weights_mean or self.args.get('weights_mean')
        std          = weights_std  or self.args.get('weights_std')
        # for uniform weights
        interval     = weights_interval or self.args.get('weights_interval')
        # for bias
        bias_init = bias_init or self.args.get('bias_init')
        # activation function!
        activation_name = activation or self.args.get('activation')
        if isinstance(activation_name, basestring):
            activation_func = get_activation_function(activation_name)
        else:
            assert callable(activation_name)
            activation_func = activation_name

        ####################################################
        # parameters - make sure to deal with params_hook! #
        ####################################################
        if params_hook:
            assert len(params_hook) == 2, "Expected 2 params (W and b) for BasicLayer, found {0!s}!".format(len(params_hook))  # make sure the params_hook has W and b
            W, b = params_hook
        else:
            # if we are initializing weights from a gaussian
            if weights_init.lower() == 'gaussian':
                W = get_weights_gaussian(shape=(input_size, output_size), mean=mean, std=std, name="W")
            # if we are initializing weights from a uniform distribution
            elif self.args.get('weights_init').lower() == 'uniform':
                W = get_weights_uniform(shape=(input_size, output_size), interval=interval, name="W")
            # otherwise not implemented
            else:
                log.error("Did not recognize weights_init %s! Pleas try gaussian or uniform" % str(self.args.get('weights_init')))
                raise NotImplementedError("Did not recognize weights_init %s! Pleas try gaussian or uniform" % str(self.args.get('weights_init')))

            b = get_bias(shape=output_size, name="b", init_values=bias_init)

        # Finally have the two parameters!
        self.params = [W, b]

        ###############
        # computation #
        ###############
        # Here is the meat of the computation transforming input -> output
        self.output = activation_func(T.dot(self.input, W) + b)

        log.debug("Initialized a basic fully-connected layer with shape %s and activation: %s" % str((input_size, output_size)), str(activation_name))

    def get_inputs(self):
        return [self.input]

    def get_outputs(self):
        return self.output

    def get_params(self):
        return self.params


class SoftmaxLayer(BasicLayer):
    """
    The softmax layer is meant as a last-step prediction layer using the softmax activation function - this class exists to provide
    easy access to methods for errors and log-likelihood for a given truth label y.

    It is a special subclass of the FullyConnectedLayer, with the activation function forced to be 'softmax'
    """
    def __init__(self, inputs_hook=None, config=None, params_hook=None, input_size=None, output_size=None,
                 weights_init=None, weights_mean=None, weights_std=None, weights_interval=None, bias_init=None):
        # init the fully connected generic layer with a softmax activation function
        super(SoftmaxLayer, self).__init__(inputs_hook=inputs_hook,
                                           params_hook=params_hook,
                                           activation='softmax',
                                           config=config,
                                           input_size=input_size,
                                           output_size=output_size,
                                           weights_init=weights_init,
                                           weights_mean=weights_mean,
                                           weights_std=weights_std,
                                           weights_interval=weights_interval,
                                           bias_init=bias_init)

        self.y_pred = T.argmax(self.get_outputs(), axis=1)

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.get_outputs())[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            log.error("y should have the same shape as self.y_pred! found y %s and y_pred %s", str(y.ndim), str(self.y_pred.ndim))
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))

        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1 represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def get_argmax_prediction(self):
        return self.y_pred