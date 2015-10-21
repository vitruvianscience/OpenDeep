"""
This module defines isotropic gaussian log-likelihood loss.
"""
# standard libraries
import logging
# third party libraries
from theano.tensor import (log as Tlog, sqrt)
from numpy import pi
# internal references
from opendeep.optimization.loss import Loss

log = logging.getLogger(__name__)


class IsotropicGaussianLL(Loss):
    """
    This takes the negative log-likelihood of an isotropic Gaussian with estimated mean and standard deviation.
    Useful for continuous-valued costs.

    .. note::
        Use this cost, for example, on Generative Stochastic Networks when the input/output is continuous
        (alternative to mse cost).
    """
    def __init__(self, inputs, targets, std_estimated):
        """
        Initializes the :class:`IsotropicGaussianLL` loss function.

        Parameters
        ----------
        inputs : theano symbolic expression
            The symbolic tensor (or compatible) representing the means of the distribution estimated.
            In the case of Generative Stochastic Networks, for example, this would be the final reconstructed output x'.
        targets : theano symbolic variable
            The symbolic tensor (or compatible) target truth to compare the means_estimated against.
        std_estimated : theano symbolic expression
            The estimated standard deviation (sigma).
        """
        self._classname = self.__class__.__name__
        log.debug("Creating a new instance of %s", self._classname)
        super(IsotropicGaussianLL, self).__init__(inputs=inputs, targets=targets, std_estimated=std_estimated)

    def get_loss(self):
        """
        Returns
        -------
        theano expression
            The loss function.
        """
        # The following definition came from the Conditional_nade project
        # the loglikelihood of isotropic Gaussian with
        # estimated mean and std
        std_estimated = self.args.get('std_estimated')
        target = self.targets[0]
        input = self.inputs[0]

        A = -((target - input) ** 2) / (2 * (std_estimated ** 2))
        B = -Tlog(std_estimated * sqrt(2 * pi))
        LL = (A + B).sum(axis=1).mean()
        return -LL

        # Example from GSN:
        # this_cost = isotropic_gaussian_LL(
        #     output=reconstruction,
        #     std_estimated=self.layers[0].sigma,
        #     target=self.inputs)
