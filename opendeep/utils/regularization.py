"""
This module provides regularization functions to help prevent overfitting in certain cases.
"""
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger", "Li Yao"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
# third party libraries
import theano.tensor as T
# internal references
from opendeep.utils.misc import raise_to_list

log = logging.getLogger(__name__)

def L1(parameters):
    """
    L1 loss, also known as square or lasso, is good for giving sparse estimates.
    Normally in practice, L2 is generally a better bet.

    Parameters
    ----------
    parameters : list of theano variables
        Parameters to apply the regularization.

    Returns
    -------
    theano expression
        L1 regularization applied to the theano variables.
    """
    # make parameters into a list if it isn't (so we can do comprehension)
    parameters = raise_to_list(parameters)
    if parameters is not None:
        return T.sum([T.sum(abs(parameter)) for parameter in parameters])
    else:
        log.warning("None parameters passed to L1 regularizer!")

def L2(parameters):
    """
    L2 loss is also known as ridge regularization (for ridge regression). It is most commonly used in practice.

    Parameters
    ----------
    parameters : list of theano variables
        Parameters to apply the regularization.

    Returns
    -------
    theano expression
        L2 regularization applied to the theano variables.
    """
    # make parameters into a list if it isn't (so we can do comprehension)
    parameters = raise_to_list(parameters)
    if parameters is not None:
        return T.sum([T.sum(parameter ** 2) for parameter in parameters])
    else:
        log.warning("None parameters passed to L2 regularizer!")

def elastic(parameters, l1_coefficient, l2_coefficient=None):
    """
    Elastic net regularization is a weighted combination of l1 and l2 regularization. This is known as
    elastic regularization.

    Parameters
    ----------
    parameters : list of theano variables
        Parameters to apply the regularization.
    l1_coefficient : float
        Weighting for L1 regularization contribution.
    l2_coefficient : float
        Weighting for L2 regularization contribution.

    Returns
    -------
    theano expression
        The appropriate regularized parameters as a weighted combination of L1 and L2.
    """
    # if the second coefficient isn't provided, just make it (1-l1_coef). Bound it at 0 though.
    if l2_coefficient is None:
        l2_coefficient = T.max(1-l1_coefficient, 0)

    if parameters is not None:
        return l1_coefficient*L1(parameters) + l2_coefficient*L2(parameters)
    else:
        log.warning("None parameters passed to elastic regularizer!")

def kl_divergence(p, q):
    """
    Kullback-Leibler divergence that is a non-symmetric measure of the difference between two probability distributions
    P and Q. It is a measure of the information lost when Q is used to approximate P. See Wikipedia.

    KL(P || Q) = p log p - p log q + (1-p) log (1-p) - (1-p) log (1-q)
    """
    term1 = p * T.log(p)
    term2 = p * T.log(q)
    term3 = (1 - p) * T.log(1 - p)
    term4 = (1 - p) * T.log(1 - q)
    return term1 - term2 + term3 - term4

def sparsity(units, sparsity_level=0.05, sparse_reg=1e-3):
    """
    Sparsity regularization for the `units`.

    .. todo:: Implement Sparsity requirement for regularization adding to the :class:`Model` cost.
    """
    raise NotImplementedError("Sparsity not implemented currently.")
    #
    # assert units.ndim == 2, "Expected units to be a matrix, but it had %d dimensions" % units.ndim
    #
    # sparsity_level = T.extra_ops.repeat(sparsity_level, units.shape[1])
    # avg_act = units.mean(axis=0)
    # kl_div = kl_divergence(sparsity_level, avg_act)
    # sparsity_penalty = sparse_reg * kl_div.sum()
    # return sparsity_penalty