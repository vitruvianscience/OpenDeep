"""
This module defines mean square error loss.
"""
# standard libraries
import logging
# third party libraries
from theano.tensor import (mean, sqr)
# internal references
from opendeep.optimization.loss import Loss

log = logging.getLogger(__name__)


class MSE(Loss):
    """
    This is the Mean Square Error (MSE) across all dimensions, or per multibatch row (depending on mean_over_second).
    """
    def __init__(self, inputs, targets, mean_over_second=True):
        """
        Initializes the :class:`MSE` loss function.

        Parameters
        ----------
        inputs : theano symbolic expression
            The symbolic tensor (or compatible) output from the network. (Comes from model).
        targets : theano symbolic variable
            The symbolic tensor (or compatible) target truth to compare the output against. (Type comes from data).
        mean_over_second : bool, optional
            Boolean whether or not to take the mean across all dimensions (True) or just the
            feature dimensions (False). Defaults to True.
        """
        super(MSE, self).__init__(inputs=inputs, targets=targets, mean_over_second=mean_over_second)

    def get_loss(self):
        """
        Returns
        -------
        theano expression
            The MSE loss function.
        """
        target = self.targets[0]
        input = self.inputs[0]
        # The following definition came from the Conditional_nade project
        if self.args.get('mean_over_second'):
            cost = mean(sqr(target - input))
        else:
            cost = mean(sqr(target - input).sum(axis=1))
        return cost
