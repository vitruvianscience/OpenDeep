"""
This module defines utilities for combining loss functions.
"""
# standard libraries
import logging
# third party libraries
from theano.tensor import (sum, mean)
# internal references
from opendeep.optimization.loss import Loss

log = logging.getLogger(__name__)


class SumLosses(Loss):
    """
    Sums a list of loss functions.
    """
    def __init__(self, inputs):
        """
        Parameters
        ----------
        inputs : list(Loss)
            The :class:`Loss` functions to add.
        """
        self._classname = self.__class__.__name__
        log.debug("Creating a new instance of %s", self._classname)
        super(SumLosses, self).__init__(inputs=inputs)

    def get_loss(self):
        """
        Returns
        -------
        theano expression
            The sum of the input loss functions
        """
        return sum(self.inputs)


class MeanLosses(Loss):
    """
    Takes the mean of a list of loss functions.
    """
    def __init__(self, inputs):
        """
        Parameters
        ----------
        inputs : list(Loss)
            The :class:`Loss` functions to average.
        """
        self._classname = self.__class__.__name__
        log.debug("Creating a new instance of %s", self._classname)
        super(MeanLosses, self).__init__(inputs=inputs)

    def get_loss(self):
        """
        Returns
        -------
        theano expression
            The mean of the input loss functions
        """
        return mean(self.inputs)
