'''
Generic structure of ADADELTA algorithm

'ADADELTA: An Adaptive Learning Rate Method'
Matthew D. Zeiler
http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
'''

__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Pylearn2","Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import logging
from collections import OrderedDict
# third party libraries
import theano.tensor as T
# internal references
from opendeep import sharedX
from opendeep.optimization.stochastic_gradient_descent import SGD
from opendeep.data.iterators.sequential import SequentialIterator

log = logging.getLogger(__name__)

# Default values to use for some training parameters
_defaults = {'decay': 0.95, # in this case, decay is rho from the paper
             "n_epoch": 1000,
             "batch_size": 100,
             "minimum_batch_size": 1,
             "save_frequency": 10,
             "early_stop_threshold": .9995,
             "early_stop_length": 30,
             "learning_rate": 1e-6, # in this case, learning_rate is epsilon from the paper
             "unsupervised": False}

# All AdaDelta needs to do is implement the get_updates() method for stochastic gradient descent
class AdaDelta(SGD):
    """
    From Pylearn2 (https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/training_algorithms/learning_rule.py)
    Implements the AdaDelta learning rule as described in:
    "AdaDelta: An Adaptive Learning Rate Method", Matthew D. Zeiler.
    Parameters
    ----------
    decay : float, optional
    Decay rate :math:`\\rho` in Algorithm 1 of the aforementioned paper.
    """
    def __init__(self, model, dataset, decay=None, iterator_class=SequentialIterator, config=None, defaults=_defaults, rng=None,
                 n_epoch=None, batch_size=None, minimum_batch_size=None, save_frequency=None,
                 early_stop_threshold=None, early_stop_length=None, learning_rate=None, flag_para_load=None):
        if not decay:
            if config:
                decay = config.get('decay', defaults.get('decay'))
            elif defaults:
                decay = defaults.get('decay')
            else:
                log.warning("AdaDelta missing 'decay' parameter in config or defaults!")
                raise AssertionError
        assert decay >= 0.
        assert decay < 1.
        self.decay = decay

        # need to call the SGD constructor after parameters are extracted because the constructor calls get_updates()!
        super(AdaDelta, self).__init__(model=model, dataset=dataset, iterator_class=iterator_class, config=config, defaults=defaults,
                                       rng=rng, n_epoch=n_epoch, batch_size=batch_size, minimum_batch_size=minimum_batch_size,
                                       save_frequency=save_frequency, early_stop_length=early_stop_length,
                                       early_stop_threshold=early_stop_threshold, learning_rate=learning_rate,
                                       flag_para_load=flag_para_load)

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
