"""
Generic implementation of ADADELTA trainig algorithm

'ADADELTA: An Adaptive Learning Rate Method'
Matthew D. Zeiler
http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
"""

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
    """
    def __init__(self, model, dataset,
                 n_epoch=10, batch_size=100, minimum_batch_size=1,
                 save_frequency=None, early_stop_threshold=None, early_stop_length=None,
                 learning_rate=1e-6, lr_decay=None, lr_factor=None,
                 decay=0.95):
        """
        Initialize AdaDelta.

        Parameters
        ----------
        model : Model
            The Model to train.
        dataset : Dataset
            The Dataset to use when training the Model.
        n_epoch : int
            how many training iterations over the dataset to go.
        batch_size : int
            How many examples from the training dataset to use in parallel.
        minimum_batch_size : int
            The minimum number of examples required at a time (for things like time series, this would be > 1).
        save_frequency : int
            How many epochs to train between each new save of the Model's parameters.
        early_stop_threshold : float
            The factor by how much the best validation training score needs to improve to determine early stopping.
        early_stop_length : int
            The patience or number of epochs to wait after the early_stop_threshold has been reached before stopping.
        learning_rate : float
            The multiplicative amount to adjust parameters based on their gradient values.
        lr_decay : str
            The type of decay function to use for changing the learning rate over epochs. See
            `opendeep.utils.decay` for options.
        lr_factor : float
            The amount to use for the decay function when changing the learning rate over epochs. See
            `opendeep.utils.decay` for its effect for given decay functions.
        decay : float
            Decay rate :math:`\\rho` in Algorithm 1 of Zeiler's paper.
        """
        # need to call the SGD constructor after parameters are extracted because the constructor calls get_updates()!
        initial_parameters = locals().copy()
        initial_parameters.pop('self')
        super(AdaDelta, self).__init__(**initial_parameters)

        assert decay >= 0., "AdaDelta Decay needs to be >=0."
        assert decay < 1., "AdaDelta Decay needs to be <1."
        self.decay = decay

    def get_updates(self, gradients):
        """
        Compute the AdaDelta updates (see the paper for details).

        Parameters
        ----------
        gradients : dict
            A dictionary mapping from the model's parameters to their
            gradients.

        Returns
        -------
        updates : OrderdDict
            A dictionary mapping from the old model parameters, to their new
            values after a single iteration of the learning rule.
        """
        log.debug('Setting up ADADELTA for optimizer...')
        updates = OrderedDict()
        for param in gradients.keys():
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
                (1 - self.decay) * T.sqr(gradients[param])
            )

            # Compute update
            epsilon = self.lr_scalers.get(param, 1.) * self.learning_rate
            rms_dx_tm1 = T.sqrt(mean_square_dx + epsilon)
            rms_grad_t = T.sqrt(new_mean_squared_grad + epsilon)
            delta_x_t = - (rms_dx_tm1 / rms_grad_t) * gradients[param]

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