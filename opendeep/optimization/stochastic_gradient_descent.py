'''
.. module:: stochastic_gradient_descent

Generic stochastic gradient descent optimization with momentum (Nesterov acceleration) and annealing.
This also serves as the base class for other learning rate update algorithms, such as ADADELTA or RMSProp.
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
# third party libraries
from theano.compat.python2x import OrderedDict  # use this compatibility OrderedDict
import theano.compat.six as six
# internal references
from opendeep import sharedX
from opendeep.optimization.optimizer import Optimizer
from opendeep.utils.decay import get_decay_function

log = logging.getLogger(__name__)

class SGD(Optimizer):
    '''
    Stochastic gradient descent for training a model - includes lr decay and momentum
    '''

    # Default values to use for some training parameters
    defaults = {"learning_rate": 0.25,
                "lr_decay": "exponential",
                "lr_factor": .995,
                "momentum": 0.5,
                'momentum_decay': 'linear',
                'momentum_factor': 0,
                'nesterov_momentum': True}

    def __init__(self, model, dataset,
                 config=None, defaults=defaults,
                 n_epoch=None, batch_size=None, minimum_batch_size=None,
                 save_frequency=None, early_stop_threshold=None, early_stop_length=None,
                 learning_rate=None, lr_decay=None, lr_factor=None,
                 momentum=None, momentum_decay=None, momentum_factor=None, nesterov_momentum=None):
        # superclass init
        super(SGD, self).__init__(model, dataset, config=config, defaults=defaults,
                                  n_epoch=n_epoch, batch_size=batch_size, minimum_batch_size=minimum_batch_size,
                                  save_frequency=save_frequency, early_stop_length=early_stop_length,
                                  early_stop_threshold=early_stop_threshold, learning_rate=learning_rate,
                                  lr_decay=lr_decay, lr_factor=lr_factor, momentum=momentum,
                                  momentum_decay=momentum_decay, momentum_factor=momentum_factor,
                                  nesterov_momentum=nesterov_momentum)
        # everything is in self! yay!

        # Momentum - smoothing over the parameter changes (see Hinton)
        if self.momentum:
            self.momentum = sharedX(self.momentum, 'momentum')
            if self.momentum_decay is not None and \
                            self.momentum_decay is not False and \
                            self.momentum_factor is not None:
                self.momentum_decay = get_decay_function(self.momentum_decay,
                                                         self.momentum,
                                                         self.momentum.get_value(),
                                                         self.momentum_factor)
            else:
                self.momentum_decay = False
        else:
            self.momentum = 1

    def get_updates(self, grads):
        """
        Based on Pylearn2
        (https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/training_algorithms/learning_rule.py)

        Implements momentum as described in Section 9 of
        "A Practical Guide to Training Restricted Boltzmann Machines",
        Geoffrey Hinton.
        Parameters are updated by the formula:
        inc := momentum * inc - learning_rate * d cost / d param
        param := param + inc

        Also has the option to implement Nesterov momentum (accelerated momentum), which works better in a lot of cases.

        :param grads: OrderedDict
        An OrderedDict of (parameter, gradient) for the model's gradients
        :return: OrderedDict
        Updates at each training step
        """
        log.debug('Setting up Stochastic Gradient Descent with momentum for optimizer...')
        updates = OrderedDict()
        for (param, gradient) in six.iteritems(grads):
            velocity = sharedX(param.get_value() * 0.)
            assert param.dtype == velocity.dtype
            assert gradient.dtype == param.dtype
            if param.name is not None:
                velocity.name = 'vel_' + param.name

            scaled_lr = self.learning_rate * self.lr_scalers.get(param, 1.)
            updates[velocity] = self.momentum * velocity - scaled_lr * gradient

            inc = updates[velocity]
            if self.nesterov_momentum:
                log.debug('Using Nesterov momentum for parameter %s', str(param))
                inc = self.momentum * inc - scaled_lr * gradient

            assert inc.dtype == velocity.dtype
            updates[param] = param + inc

        return updates