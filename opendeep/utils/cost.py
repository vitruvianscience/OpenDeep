"""
.. module:: cost

These functions are used as the objectives (costs) to minimize during training of deep networks.
You should be careful to use the appropriate cost function for the type of input and output of the network.

EVERY COST FUNCTION SHOULD INCLUDE AND OUTPUT AND TARGET PARAMETER. Extra parameters can be included and named
whatever you like.
"""
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
# third party libraries
import theano.tensor as T
import numpy

log = logging.getLogger(__name__)

def binary_crossentropy(output, target):
    """
    Computes the mean binary cross-entropy between a target and an output, across all dimensions
    (both the feature and example dimensions).

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

def categorical_crossentropy(output, target):
    """
    This is the mean multinomial negative log-loss.
    From Theano:
    Return the mean cross-entropy between an approximating distribution and a true distribution, across all dimensions.
    The cross entropy between two probability distributions measures the average number of bits needed to identify an
    event from a set of possibilities, if a coding scheme is used based on a given probability distribution q, rather
    than the "true" distribution p.
    Mathematically, this function computes H(p,q) = - \sum_x p(x) \log(q(x)), where p=target_distribution and
    q=coding_distribution.

    :param output: symbolic 2D tensor (or compatible) where each row represents a distribution
    :type output: Tensor

    :param target: symbolic 2D tensor *or* symbolic vector of ints. In the case of an integer vector argument,
    each element represents the position of the '1' in a 1-of-N encoding (aka 'one-hot' encoding)
    :type target: Tensor

    :return: the mean of the cross-entropy tensor
    :rtype: Tensor
    """
    return T.mean(T.nnet.categorical_crossentropy(output, target))

def mse(output, target, mean_over_second=True):
    """
    This is the Mean Square Error (MSE) across all dimensions, or per multibatch row (depending on mean_over_second).

    :param output: the symbolic tensor (or compatible) output from the network
    :type output: Tensor

    :param target: the symbolic tensor (or compatible) target truth to compare the output against.
    :type target: Tensor

    :param mean_over_second: boolean whether or not to take the mean across all dimensions (True) or just the
    feature dimensions (False)
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
def isotropic_gaussian_LL(output, target, std_estimated):
    """
    This takes the negative log-likelihood of an isotropic Gaussian with estimated mean and standard deviation.
    Useful for continuous-valued costs.

    :param output: the symbolic tensor (or compatible) representing the means of the distribution estimated.
    In the case of Generative Stochastic Networks, for example, this would be the final reconstructed output x'.
    :type output: Tensor

    :param std_estimated: the estimated standard deviation (sigma)
    :type std_estimated: Tensor

    :param target: the symbolic tensor (or compatible) target truth to compare the means_estimated against.
    :type target: Tensor

    :return: the negative log-likelihood
    :rtype: Tensor

    :note: Use this cost, for example, on Generative Stochastic Networks when the input/output is continuous
    (alternative to mse cost).
    """
    # The following definition came from the Conditional_nade project
    #the loglikelihood of isotropic Gaussian with
    # estimated mean and std
    A = -((target - output)**2) / (2*(std_estimated**2))
    B = -T.log(std_estimated * T.sqrt(2*numpy.pi))
    LL = (A + B).sum(axis=1).mean()
    return -LL
    # Example from GSN:
    # this_cost = isotropic_gaussian_LL(
    #     output=reconstruction,
    #     std_estimated=self.layers[0].sigma,
    #     target=self.inputs)


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
    'zero_one': zero_one,
    'nll': mse  # this is used as a placeholder - negative log-likelihood will be taken care of by the class.
}

def get_cost_function(name):
    """
        This helper method returns the appropriate cost function given a string name. It looks up the appropriate
        function from the internal _functions dictionary.

        :param name: String representation of the cost function you want (normally grabbed from a config file)
        :type name: String

        :return: The appropriate cost function, or raise NotImplementedError if it isn't found.
        :rtype: Method

        :raises: NotImplementedError
        """
    # if the name is callable, return the function
    if callable(name):
        return name
    # otherwise if it is a string name
    elif isinstance(name, basestring):
        # standardize the input to be lowercase
        name = name.lower()
        # grab the appropriate activation function from the dictionary of functions
        func = _functions.get(name)
        # if it couldn't find the function (key didn't exist), raise a NotImplementedError
        if func is None:
            log.error("Did not recognize cost function %s! Please use one of: ", str(name), str(_functions.keys()))
            raise NotImplementedError(
                "Did not recognize cost function {0!s}! Please use one of: {1!s}".format(name, _functions.keys())
            )
        # return the found function
        return func
    # otherwise we don't know what to do.
    else:
        log.error("Cost function not implemented for %s with type %s", str(name), str(type(name)))
        raise NotImplementedError("Cost function not implemented for %s with type %s", str(name), str(type(name)))