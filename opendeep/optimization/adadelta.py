'''
.. module:: adadelta

Generic implementation of ADADELTA trainig algorithm

'ADADELTA: An Adaptive Learning Rate Method'
Matthew D. Zeiler
http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
'''

__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Pylearn2", "Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
from collections import OrderedDict
# third party libraries
import theano.tensor as T
# internal references
from opendeep import sharedX
from opendeep.optimization.optimizer import Optimizer

log = logging.getLogger(__name__)

# All AdaDelta needs to do is implement the get_updates() method for stochastic gradient descent
class AdaDelta(Optimizer):
    """
    From Pylearn2 (https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/training_algorithms/learning_rule.py)
    Implements the AdaDelta learning rule as described in:
    "AdaDelta: An Adaptive Learning Rate Method", Matthew D. Zeiler.
    Parameters
    ----------
    decay : float, optional
    Decay rate :math:`\\rho` in Algorithm 1 of the aforementioned paper.
    """

    # Default values to use for some training parameters
    defaults = {'decay': 0.95,  # rho
                'learning_rate': 1e-6  # epsilon
                }

    def __init__(self, model, dataset,
                 config=None, defaults=defaults,
                 n_epoch=None, batch_size=None, minimum_batch_size=None,
                 save_frequency=None, early_stop_threshold=None, early_stop_length=None,
                 learning_rate=None, lr_decay=None, lr_factor=None,
                 decay=None):
        # need to call the SGD constructor after parameters are extracted because the constructor calls get_updates()!
        super(AdaDelta, self).__init__(model, dataset, config=config, defaults=defaults,
                                       n_epoch=n_epoch, batch_size=batch_size, minimum_batch_size=minimum_batch_size,
                                       save_frequency=save_frequency, early_stop_length=early_stop_length,
                                       early_stop_threshold=early_stop_threshold, learning_rate=learning_rate,
                                       lr_decay=lr_decay, lr_factor=lr_factor, decay=decay)

        assert self.decay >= 0., "Decay needs to be >=0."
        assert self.decay < 1., "Decay needs to be <1."

    def get_updates(self, grads):
        """
        Compute the AdaDelta updates
        Parameters
        ----------
        learning_rate : float
        Learning rate coefficient.
        grads : dict
        A dictionary mapping from the model's parameters to their
        gradients.
        """
        log.debug('Setting up ADADELTA for optimizer...')
        updates = OrderedDict()
        for param in grads.keys():
            # mean_squared_grad := E[g^2]_{t-1}
            mean_square_grad = sharedX(param.get_value() * 0.)
            # mean_square_dx := E[(\Delta x)^2]_{t-1}
            mean_square_dx = sharedX(param.get_value() * 0.)

            if param.name is not None:
                mean_square_grad.name = 'mean_square_grad_' + param.name
                mean_square_dx.name = 'mean_square_dx_' + param.name

            # Accumulate gradient
            new_mean_squared_grad = (
                self.decay * mean_square_grad +
                (1 - self.decay) * T.sqr(grads[param])
            )

            # Compute update
            epsilon = self.lr_scalers.get(param, 1.) * self.learning_rate
            rms_dx_tm1 = T.sqrt(mean_square_dx + epsilon)
            rms_grad_t = T.sqrt(new_mean_squared_grad + epsilon)
            delta_x_t = - (rms_dx_tm1 / rms_grad_t) * grads[param]

            # Accumulate updates
            new_mean_square_dx = (
                self.decay * mean_square_dx +
                (1 - self.decay) * T.sqr(delta_x_t)
            )

            # Apply update
            updates[mean_square_grad] = new_mean_squared_grad
            updates[mean_square_dx] = new_mean_square_dx
            updates[param] = param + delta_x_t

        return updates
