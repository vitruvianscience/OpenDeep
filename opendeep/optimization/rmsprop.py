'''
Generic structure of RMSProp algorithm
'''

__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Pylearn2", "Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import logging
# third party libraries
import theano.tensor as T
from theano.compat.python2x import OrderedDict  # use this compatibility OrderedDict
# internal references
from opendeep import sharedX
from opendeep.optimization.stochastic_gradient_descent import SGD
from opendeep.data.iterators.sequential import SequentialIterator

log = logging.getLogger(__name__)

# Default values to use for some training parameters
_defaults = {'decay': 0.95,
             'max_scaling': 1e5,
             "n_epoch": 1000,
             "batch_size": 100,
             "minimum_batch_size": 1,
             "save_frequency": 10,
             "early_stop_threshold": .9995,
             "early_stop_length": 30,
             "learning_rate": 1e-6,
             "unsupervised": False}

# All RMSProp needs to do is implement the get_updates() method for stochastic gradient descent
class RMSProp(SGD):
    """
    From Pylearn2 (https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/training_algorithms/learning_rule.py)

    The RMSProp learning rule is described by Hinton in `lecture 6
    <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`
    of the Coursera Neural Networks for Machine Learning course.
    In short, Hinton suggests "[the] magnitude of the gradient can be very
    different for different weights and can change during learning. This
    makes it hard to choose a global learning rate." RMSProp solves this
    problem by "[dividing] the learning rate for a weight by a running
    average of the magnitudes of recent gradients for that weight."
    Parameters
    ----------
    decay : float, optional
    Decay constant similar to that used in AdaDelta and Momentum methods.
    max_scaling: float, optional
    Restrict the RMSProp gradient scaling coefficient to values
    below `max_scaling`.
    """
    def __init__(self, model, dataset, decay=None, max_scaling=None, iterator_class=SequentialIterator, config=None, defaults=_defaults,
                 rng=None, n_epoch=None, batch_size=None, minimum_batch_size=None, save_frequency=None,
                 early_stop_threshold=None, early_stop_length=None, learning_rate=None, flag_para_load=None):
        if not decay:
            if config:
                decay = config.get('decay', defaults.get('decay'))
            elif defaults:
                decay = defaults.get('decay')
            else:
                log.error("RMSProp missing 'decay' parameter in config or defaults!")
                raise AssertionError
        assert decay >= 0.
        assert decay < 1.
        self.decay = sharedX(decay)

        if not max_scaling:
            if config:
                max_scaling = config.get('max_scaling', defaults.get('max_scaling'))
            elif defaults:
                max_scaling = defaults.get('max_scaling')
            else:
                log.error("RMSProp missing 'max_scaling' parameter in config or defaults!")
                raise AssertionError
        assert max_scaling > 0.
        self.epsilon = 1. / max_scaling

        self.mean_square_grads = OrderedDict()

        # need to call the SGD constructor after parameters are extracted because the constructor calls get_updates()!
        super(RMSProp, self).__init__(model=model, dataset=dataset, iterator_class=iterator_class, config=config, defaults=defaults,
                                      rng=rng, n_epoch=n_epoch, batch_size=batch_size, minimum_batch_size=minimum_batch_size,
                                      save_frequency=save_frequency, early_stop_length=early_stop_length,
                                      early_stop_threshold=early_stop_threshold, learning_rate=learning_rate,
                                      flag_para_load=flag_para_load)

    def get_updates(self, grads):
        """
        Provides the symbolic (theano) description of the updates needed to
        perform this learning rule. See Notes for side-effects.

        Parameters
        ----------
        grads : dict
            A dictionary mapping from the model's parameters to their
            gradients.

        Returns
        -------
        updates : OrderdDict
            A dictionary mapping from the old model parameters, to their new
            values after a single iteration of the learning rule.

        Notes
        -----
        This method has the side effect of storing the moving average
        of the square gradient in `self.mean_square_grads`. This is
        necessary in order for the monitoring channels to be able
        to track the value of these moving averages.
        Therefore, this method should only get called once for each
        instance of RMSProp.
        """
        log.debug('Setting up RMSProp for optimizer...')
        updates = OrderedDict()
        for param in grads:

            # mean_squared_grad := E[g^2]_{t-1}
            mean_square_grad = sharedX(param.get_value() * 0.)

            if param.name is None:
                raise ValueError("Model parameters must be named.")
            mean_square_grad.name = 'mean_square_grad_' + param.name

            if param.name in self.mean_square_grads:
                log.warning("Calling get_updates more than once on the "
                              "gradients of `%s` may make monitored values "
                              "incorrect." % param.name)
            # Store variable in self.mean_square_grads for monitoring.
            self.mean_square_grads[param.name] = mean_square_grad

            # Accumulate gradient
            new_mean_squared_grad = (self.decay * mean_square_grad +
                                     (1 - self.decay) * T.sqr(grads[param]))

            # Compute update
            scaled_lr = self.lr_scalers.get(param, 1.) * self.learning_rate
            rms_grad_t = T.sqrt(new_mean_squared_grad)
            rms_grad_t = T.maximum(rms_grad_t, self.epsilon)
            delta_x_t = - scaled_lr * grads[param] / rms_grad_t

            # Apply update
            updates[mean_square_grad] = new_mean_squared_grad
            updates[param] = param + delta_x_t

        return updates

