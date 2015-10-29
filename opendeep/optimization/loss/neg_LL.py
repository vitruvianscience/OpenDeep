"""
This module defines negative log-likelihood loss.
"""
# standard libraries
import logging
# third party libraries
from theano.tensor import (mean, log as Tlog, arange)
# internal references
from opendeep.optimization.loss import Loss

log = logging.getLogger(__name__)


class Neg_LL(Loss):
    """
    Defines the mean of the negative log-likelihood of the prediction
    of this model under a given target distribution.

    Notes
    -----
    We use the mean instead of the sum so that the learning rate is less dependent on the batch size.
    TARGETS MUST BE ONE-HOT ENCODED (a vector with 0's except 1 for the correct label).
    """
    def __init__(self, inputs, targets, one_hot=True):
        """
        Initializes the :class:`ZeroOne` loss function.

        Parameters
        ----------
        inputs : theano symbolic expression
            The output probability of target given input P(Y|X).
        targets : theano symbolic variable
            The correct target labels Y.
        one_hot : bool
            Whether the label targets Y are encoded as a one-hot vector or as the int class label.
            If it is not one-hot, needs to be 2-dimensional.
        """
        super(Neg_LL, self).__init__(inputs=inputs, targets=targets, one_hot=one_hot)

    def get_loss(self):
        """
        Returns
        -------
        theano expression
            The negative log-likelihood loss between inputs and targets.
        """
        p_y_given_x = self.inputs[0]
        y = self.targets[0]
        # y.shape[0] is (symbolically) the number of examples (call it n) in the minibatch.
        # T.arange(y.shape[0]) is a symbolic vector which will contain [0,1,2,... n-1]
        # T.log(self.p_y_given_x) is a matrix of Log-Probabilities (call it LP) with one row per example and
        # one column per class
        # LP[T.arange(y.shape[0]),y] is a vector v containing
        # [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ..., LP[n-1,y[n-1]]] and
        # T.mean(LP[T.arange(y.shape[0]),y]) is the mean (across minibatch examples) of the elements in v,
        # i.e. the mean log-likelihood across the minibatch.
        if self.args.get('one_hot'):
            # if one_hot, labels y act as a mask over p_y_given_x
            assert y.ndim == p_y_given_x.ndim, "Need to have target same dimensions as model output, found %d and %d" \
                % (y.ndim, p_y_given_x.ndim)
            return -mean(Tlog(p_y_given_x) * y)
        else:
            assert p_y_given_x.ndim == 2, "Need to have 2D model output, found %d" % p_y_given_x.ndim
            assert y.ndim == 1
            return -mean(Tlog(p_y_given_x)[arange(y.shape[0]), y])
