"""
This module defines expressions for finding the loss or cost function for models to optimize.
"""
# standard libraries
import logging
# internal references
from opendeep.utils.misc import raise_to_list

log = logging.getLogger(__name__)


class Loss(object):
    """
    The :class:`Loss` class takes a Theano expression and a target Theano symbolic variable to compute
    the loss function.

    Attributes
    ----------
    inputs : list
        List of theano symbolic expressions that are the necessary inputs to the loss function.
    targets : list
        List of target theano symbolic variables (or empty list) necessary for the loss function.
    args : dict
        Dictionary of all parameter arguments to the class initialization.
    """
    def __init__(self, inputs, targets=None, func=None, **kwargs):
        """
        Initializes the :class:`Loss` function.

        Parameters
        ----------
        inputs : list(theano symbolic expression)
            The input(s) necessary for the loss function.
        targets : list(theano symbolic variable), optional
            The target(s) variables for the loss function.
        func : function, optional
            A python function for computing the loss given the inputs list an targets list (in order).
            The function `func` will be called with parameters: func(*(list(inputs)+list(targets))).
        """
        self._classname = self.__class__.__name__
        log.debug("Creating a new instance of %s", self._classname)
        self.inputs = raise_to_list(inputs)
        self.targets = raise_to_list(targets) or []
        self.func = func
        self.args = kwargs.copy()
        self.args['inputs'] = self.inputs
        self.args['targets'] = self.targets
        self.args['func'] = self.func

    def get_loss(self):
        """
        Returns the expression for the loss function.

        Returns
        -------
        theano expression
            The loss function.
        """
        if self.func is not None:
            inputs_targets = self.inputs + self.targets
            return self.func(*inputs_targets)
        else:
            raise NotImplementedError("Loss function not defined for %s" % self._classname)

    def get_targets(self):
        """
        Returns the target(s) Theano symbolic variables used to compute the loss. These will be fed
        into the training function and should have the right dtype for the intended data.

        Returns
        -------
        symbolic variable or list(theano symbolic variable)
            The symbolic variable(s) used when computing the loss (the target values). By default, this returns
            the `targets` parameter when initializing the :class:`Loss` class, raised to a list.
        """
        return self.targets
