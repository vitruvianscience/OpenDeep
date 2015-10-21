"""
This module defines categorical crossentropy loss.
"""
# standard libraries
import logging
# third party libraries
from theano.tensor import (mean, nnet)
# internal references
from opendeep.optimization.loss import Loss

log = logging.getLogger(__name__)


class CategoricalCrossentropy(Loss):
    """
    This is the mean multinomial negative log-loss.

    From Theano:
    Return the mean cross-entropy between an approximating distribution and a true distribution, across all dimensions.
    The cross entropy between two probability distributions measures the average number of bits needed to identify an
    event from a set of possibilities, if a coding scheme is used based on a given probability distribution q, rather
    than the "true" distribution p.
    Mathematically, this function computes H(p,q) = - \sum_x p(x) \log(q(x)), where p=target_distribution and
    q=coding_distribution.
    """
    def __init__(self, inputs, targets):
        """
        Initializes the :class:`CategoricalCrossentropy` loss function.

        Parameters
        ----------
        inputs : theano symbolic expression
            Symbolic 2D tensor (or compatible) where each row represents a distribution.
        targets : theano symbolic variable
            Symbolic 2D tensor *or* symbolic vector of ints. In the case of an integer vector argument,
            each element represents the position of the '1' in a 1-of-N encoding (aka 'one-hot' encoding)
        """
        self._classname = self.__class__.__name__
        log.debug("Creating a new instance of %s", self._classname)
        super(CategoricalCrossentropy, self).__init__(inputs=inputs, targets=targets)

    def get_loss(self):
        """
        The mean of the categorical cross-entropy tensor.

        Returns
        -------
        theano expression
            The loss function.
        """
        input = self.inputs[0]
        target = self.targets[0]
        return mean(nnet.categorical_crossentropy(input, target))
