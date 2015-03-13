"""
.. module:: cost_functions

These functions are used as the objectives (costs) to minimize during training of deep networks. You should be careful to use
the appropriate cost function for the type of input and output of the network.
"""
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import logging
# third party libraries
import theano.tensor as T
import numpy

log = logging.getLogger(__name__)

def binary_crossentropy(output, target):
    """
    Computes the mean binary cross-entropy between a target and an output, across all dimensions (both the feature and example dimensions).

    :param output: symbolic Tensor (or compatible) that is your output from the network
    :type output: Tensor

    :param target: symbolic Tensor (or compatible) that is the target truth you want to compare the output against.
    :type target: Tensor

    :return: the mean of the binary cross-entropy tensor, where binary cross-entropy is applied element-wise:
            crossentropy(target,output) = -(target*log(output) + (1 - target)*(log(1 - output)))
    :rtype: Tensor

    :note: Use this cost for binary outputs, like MNIST.
    """
    # return T.mean(T.nnet.binary_crossentropy(output, target))
    # The following definition came from the Conditional_nade project
    L = - T.mean(target * T.log(output) +
                 (1 - target) * T.log(1 - output), axis=1)
    cost = T.mean(L)
    return cost

def categorical_crossentropy(output_dist, target_dist):
    """
    This is the mean multinomial negative log-loss.
    From Theano:
    Return the mean cross-entropy between an approximating distribution and a true distribution, across all dimensions.
    The cross entropy between two probability distributions measures the average number of bits needed to identify an event from a set
    of possibilities, if a coding scheme is used based on a given probability distribution q, rather than the "true" distribution p.
    Mathematically, this function computes H(p,q) = - \sum_x p(x) \log(q(x)), where p=target_distribution and q=coding_distribution.

    :param output: symbolic 2D tensor (or compatible) where each row represents a distribution
    :type output: Tensor

    :param target: symbolic 2D tensor *or* symbolic vector of ints. In the case of an integer vector argument, each element represents
    the position of the '1' in a 1-of-N encoding (aka 'one-hot' encoding)
    :type target: Tensor

    :return: the mean of the cross-entropy tensor
    :rtype: Tensor
    """
    return T.mean(T.nnet.categorical_crossentropy(output_dist, target_dist))

def mse(output, target, mean_over_second=True):
    """
    This is the Mean Square Error (MSE) across all dimensions, or per multibatch row (depending on mean_over_second).

    :param output: the symbolic tensor (or compatible) output from the network
    :type output: Tensor

    :param target: the symbolic tensor (or compatible) target truth to compare the output against.
    :type target: Tensor

    :param mean_over_second: boolean whether or not to take the mean across all dimensions (True) or just the feature dimensions (False)
    :type mean_over_second: Boolean

    :return: the appropriate mean square error
    :rtype: Tensor
    """
    # The following definition came from the Conditional_nade project
    if mean_over_second:
        cost = T.mean(T.sqr(target - output))
    else:
        cost = T.mean(T.sqr(target - output).sum(axis=1))
    return cost


# use this for continuous inputs
def isotropic_gaussian_LL(means_estimated, stds_estimated, targets):
    """
    This takes the negative log-likelihood of an isotropic Gaussian with estimated mean and standard deviation. Useful for continuous-valued
    costs.

    :param means_estimated: the symbolic tensor (or compatible) representing the means of the distribution estimated.
    In the case of Generative Stochastic Networks, for example, this would be the final reconstructed output x'.
    :type means_estimated: Tensor

    :param stds_estimated: the estimated standard deviation (sigma)
    :type stds_estimated: Tensor

    :param targets: the symbolic tensor (or compatible) target truth to compare the means_estimated against.
    :type targets: Tensor

    :return: the negative log-likelihood
    :rtype: Tensor

    :note: Use this cost, for example, on Generative Stochastic Networks when the input/output is continuous (alternative to mse cost).
    """
    # The following definition came from the Conditional_nade project
    #the loglikelihood of isotropic Gaussian with
    # estimated mean and std
    A = -((targets - means_estimated)**2) / (2*(stds_estimated**2))
    B = -T.log(stds_estimated * T.sqrt(2*numpy.pi))
    LL = (A + B).sum(axis=1).mean()
    return -LL
    # this_cost = isotropic_gaussian_LL(
    #     means_estimated=reconstruction,
    #     stds_estimated=self.layers[0].sigma,
    #     targets=self.inputs)


def zero_one(output, target):
    """
    This defines the zero-one loss function, where the loss is equal to the number of incorrect estimations.

    :param output: the estimated variable
    :type output: theano tensor

    :param target: the ground truth
    :type target: theano tensor

    :return: the appropriate zero-one loss
    :rtype: theano tensor
    """
    return T.sum(T.neq(output, target))


########### keep cost functions above this line, and add them to the dictionary below ####################
_functions = {
    'binary_crossentropy': binary_crossentropy,
    'categorical_crossentropy': categorical_crossentropy,
    'mse': mse,
    'isotropic_gaussian': isotropic_gaussian_LL,
    'zero_one': zero_one
}

def get_cost_function(name):
    """
        This helper method returns the appropriate cost function given a string name. It looks up the appropriate function from the
        internal _functions dictionary.

        :param name: String representation of the cost function you want (normally grabbed from a config file)
        :type name: String

        :return: The appropriate cost function, or raise NotImplementedError if it isn't found.
        :rtype: Method

        :raises: NotImplementedError
        """
    # standardize the input to be lowercase
    name = name.lower()
    # grab the appropriate activation function from the dictionary of functions
    func = _functions.get(name)
    # if it couldn't find the function (key didn't exist), raise a NotImplementedError
    if func is None:
        log.critical("Did not recognize cost function %s! Please use one of: ", str(name), str(_functions.keys()))
        raise NotImplementedError("Did not recognize cost function {0!s}! Please use one of: {1!s}".format(name, _functions.keys()))
    # return the found function
    return func