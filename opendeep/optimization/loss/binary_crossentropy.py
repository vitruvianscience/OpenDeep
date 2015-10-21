"""
This module defines binary crossentropy loss.
"""
# standard libraries
import logging
# third party libraries
from theano.tensor import (mean, nnet)
# internal references
from opendeep.optimization.loss import Loss

log = logging.getLogger(__name__)


class BinaryCrossentropy(Loss):
    """
    Computes the mean binary cross-entropy between a target and an output, across all dimensions
    (both the feature and example dimensions).

    .. note::
        Use this cost for binary outputs, like MNIST.

    """
    def __init__(self, inputs, targets):
        """
        Initializes the :class:`BinaryCrossentropy` loss function.

        Parameters
        ----------
        inputs : theano symbolic expression
            The input necessary for the loss function. Comes from Model.
        targets : theano symbolic variable
            The target variables for the loss function.
        """
        self._classname = self.__class__.__name__
        log.debug("Creating a new instance of %s", self._classname)
        super(BinaryCrossentropy, self).__init__(inputs=inputs, targets=targets)

    def get_loss(self):
        """
        The mean of the binary cross-entropy tensor, where binary cross-entropy is applied element-wise:
        crossentropy(target,input) = -(target*log(input) + (1 - target)*(log(1 - input))).

        Returns
        -------
        theano expression
            The loss function.
        """
        input = self.inputs[0]
        target = self.targets[0]
        return mean(nnet.binary_crossentropy(input, target))
        # The following definition came from the Conditional_nade project
        # L = - T.mean(target * T.log(output) +
        #              (1 - target) * T.log(1 - output), axis=1)
        # cost = T.mean(L)
        # return cost
