"""
.. module:: regularization

This module provides regularization functions to help prevent overfitting in certain cases.
"""
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger", "Li Yao"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import logging
# third party libraries
import theano.tensor as T
# internal references
from opendeep.utils.misc import raise_to_list

log = logging.getLogger(__name__)

def L1(parameters):
    """
    L1 loss, also known as square or lasso, is good for giving sparse estimates. Normally in practice, L2 is generally a better bet.

    :param parameters: parameters to apply the regularization
    :type parameters: theano variables

    :return: L1 applies to the theano variables
    :rtype: theano tensor
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

    :param parameters: parameters to apply the regularization
    :type parameters: theano variables

    :return: L1 applies to the theano variables
    :rtype: theano tensor
    """
    # make parameters into a list if it isn't (so we can do comprehension)
    parameters = raise_to_list(parameters)
    if parameters is not None:
        return T.sum([T.sum(parameter ** 2) for parameter in parameters])
    else:
        log.warning("None parameters passed to L2 regularizer!")

def elastic(parameters, l1_coefficient, l2_coefficient=None):
    """
    Elastic net regularization is a weighted combination of l1 and l2 regularization.

    :param parameters: parameters to apply the regularization
    :type parameters: theano variables

    :param l1_coefficient: weighting for l1 regularization
    :type l1_coefficient: float

    :param l2_coefficient: weighting for l2 regularization
    :type l2_coefficient: float

    :return: the appropriate regularized parameters
    :rtype: theano tensor
    """
    # if the second coefficient isn't provided, just make it (1-l1_coef). Bound it at 0 though.
    if l2_coefficient is None:
        l2_coefficient = T.max(1-l1_coefficient, 0)

    if parameters is not None:
        return l1_coefficient*L1(parameters) + l2_coefficient*L2(parameters)
    else:
        log.warning("None parameters passed to elastic regularizer!")