"""
Generic implementation of RMSProp training algorithm.
"""
# standard libraries
import logging
# third party libraries
import theano.tensor as T
from theano.compat.python2x import OrderedDict  # use this compatibility OrderedDict
# internal references
from opendeep.utils.constructors import sharedX
from opendeep.optimization.optimizer import Optimizer

log = logging.getLogger(__name__)


# All RMSProp needs to do is implement the get_updates() method for stochastic gradient descent
class RMSProp(Optimizer):
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
    """
    def __init__(self, dataset, model=None,
                 epochs=10, batch_size=100, min_batch_size=1,
                 save_freq=None, stop_threshold=None, stop_patience=None,
                 learning_rate=1e-6, lr_decay=None, lr_decay_factor=None,
                 decay=0.95, max_scaling=1e5,
                 grad_clip=None, hard_clip=False):
        """
        Initialize RMSProp.

        Parameters
        ----------
        dataset : Dataset
            The Dataset to use when training the Model.
        model : Model
            The Model to train. Needed if the Optimizer isn't being passed to a Model's .train() method.
        epochs : int
            how many training iterations over the dataset to go.
        batch_size : int
            How many examples from the training dataset to use in parallel.
        min_batch_size : int
            The minimum number of examples required at a time (for things like time series, this would be > 1).
        save_freq : int
            How many epochs to train between each new save of the Model's parameters.
        stop_threshold : float
            The factor by how much the best validation training score needs to improve to determine early stopping.
        stop_patience : int
            The patience or number of epochs to wait after the stop_threshold has been reached before stopping.
        learning_rate : float
            The multiplicative amount to adjust parameters based on their gradient values.
        lr_decay : str
            The type of decay function to use for changing the learning rate over epochs. See
            `opendeep.utils.decay` for options.
        lr_decay_factor : float
            The amount to use for the decay function when changing the learning rate over epochs. See
            `opendeep.utils.decay` for its effect for given decay functions.
        decay : float, optional
            Decay constant similar to that used in AdaDelta and Momentum methods.
        max_scaling: float, optional
            Restrict the RMSProp gradient scaling coefficient to values
            below `max_scaling`.
        grad_clip : float, optional
            Whether to clip gradients. This will clip with a maximum of grad_clip or the parameter norm.
        hard_clip : bool
            Whether to use a hard cutoff or rescaling for clipping gradients.
        """
        # need to call the Optimizer constructor
        initial_parameters = locals().copy()
        initial_parameters.pop('self')
        super(RMSProp, self).__init__(**initial_parameters)

        assert max_scaling > 0., "Max_scaling needs to be > 0."
        self.max_scaling = max_scaling
        self.epsilon = 1. / self.max_scaling
        self.decay = decay
        self.mean_square_grads = OrderedDict()

    def get_updates(self, gradients):
        """
        Provides the symbolic (theano) description of the updates needed to
        perform this learning rule. See Notes for side-effects.

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
        for param in gradients:

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
                                     (1 - self.decay) * T.sqr(gradients[param]))

            # Compute update
            scaled_lr = self.lr_scalers.get(param, 1.) * self.learning_rate
            rms_grad_t = T.sqrt(new_mean_squared_grad)
            rms_grad_t = T.maximum(rms_grad_t, self.epsilon)
            delta_x_t = - scaled_lr * gradients[param] / rms_grad_t

            # Apply update
            updates[mean_square_grad] = new_mean_squared_grad
            updates[param] = param + delta_x_t

        return updates