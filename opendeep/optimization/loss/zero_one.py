"""
This module defines zero-one loss.
"""
# standard libraries
import logging
# third party libraries
from theano.tensor import (sum, neq)
# internal references
from opendeep.optimization.loss import Loss

log = logging.getLogger(__name__)


class ZeroOne(Loss):
    """
    This defines the zero-one loss function, where the loss is equal to the number of incorrect estimations.
    """
    def __init__(self, inputs, targets):
        """
        Initializes the :class:`ZeroOne` loss function.

        Parameters
        ----------
        inputs : theano symbolic expression
            The estimated variable. (Output from computation).
        targets : theano symbolic variable
            The ground truth variable. (Type comes from data).
        """
        super(ZeroOne, self).__init__(inputs=inputs, targets=targets)

    def get_loss(self):
        """
        Returns
        -------
        theano expression
            The appropriate zero-one loss between inputs and targets.
        """
        input = self.inputs[0]
        target = self.targets[0]
        return sum(neq(input, target))
