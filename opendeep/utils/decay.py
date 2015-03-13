"""
.. module:: decay_functions

Functions used for decaying Theano parameters as shared variables.
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
import numpy
# internal references
from opendeep import cast32

log = logging.getLogger(__name__)

class DecayFunction(object):
    """
    Interface for a parameter decay function (like learning rate, noise levels, etc.)
    """
    def __init__(self, param, initial, reduction_factor):
        """
        A generic class for decaying a theano variable.

        :param param: the theano variable you want to decay. This must already be a shared variable.
        :type param: shared variable

        :param initial: the initial value the variable should have
        :type initial: Float

        :param reduction_factor: the amount of reduction (depending on subclass's algorithm) each epoch
        :type reduction_factor: Float
        """
        # make sure the parameter is a Theano shared variable
        if not hasattr(param, 'get_value'):
            log.error('Parameter doesn\'t have a get_value() function! It is supposed to be a shared variable...')
        if not hasattr(param, 'set_value'):
            log.error('Parameter doesn\'t have a set_value() function! It is supposed to be a shared variable...')
        assert hasattr(param, 'get_value')
        assert hasattr(param, 'set_value')

        self.param = param
        self.initial = initial
        self.param.set_value(cast32(self.initial))
        self.reduction_factor = reduction_factor

    def decay(self):
        """
        This will decay the shared variable according to the decay rule of the subclass.

        :return: Nothing, since decaying a shared variable modifies the memory.
        """
        log.critical('Parameter decay function %s does not have a decay method!', str(type(self)))
        raise NotImplementedError()

    def reset(self):
        """
        Resets this shared variable to the initial value provided during the constructor.

        :return: Nothing, since setting a shared variable's value modifies the memory.
        """
        self.param.set_value(self.initial)

    def simulate(self, initial, reduction_factor, epoch):
        """
        This will take an initial value for a hypothetical variable, the reduction factor appropriate to the subclass's decay function,
        and the number of decays (epoch) you want to see the simulated result after.

        :param initial: initial value for the variable
        :type initial: Float

        :param reduction_factor: the appropriate reduction factor parameter (for the subclass)
        :type reduction_factor: Float

        :param epoch: number of timesteps to simulate
        :type epoch: Integer

        :return: the simulated value depending on the reduction after the given number of epochs
        :rtype: Float
        """
        log.critical('Parameter decay function %s does not have a simulate method!', str(type(self)))
        raise NotImplementedError()


class Linear(DecayFunction):
    """
    Class for a monotonically decreasing parameter, bottoming out at 0.
    The decay function is described as:
    value = max((initial_value - reduction_factor * number_of_timesteps), 0)
    """
    def __init__(self, param, initial, reduction_factor):
        super(self.__class__, self).__init__(param, initial, reduction_factor)
        if self.reduction_factor is None:
            self.reduction_factor = 0

    def decay(self):
        new_value = self.param.get_value() - self.reduction_factor
        self.param.set_value(cast32(numpy.max([0, new_value])))

    def simulate(self, initial_value, reduction_factor, epoch):
        new_value = initial_value - reduction_factor*epoch
        return numpy.max([0, new_value])


class Exponential(DecayFunction):
    """
    Class for exponentially decreasing parameter.
    Decay function described as:
    value = initial_value * reduction_factor^number_of_timesteps
    """
    def __init__(self, param, initial, reduction_factor):
        super(self.__class__, self).__init__(param, initial, reduction_factor)
        if self.reduction_factor is None:
            self.reduction_factor = 1

    def decay(self):
        new_value = self.param.get_value()*self.reduction_factor
        self.param.set_value(cast32(new_value))

    def simulate(self, initial_value, reduction_factor, epoch):
        new_value = initial_value*pow(reduction_factor, epoch)
        return new_value


class Montreal(DecayFunction):
    """
    Class for decreasing a parameter as used by some people in the LISA lab at University of Montreal.
    Decay function described as:
    value = initial_value / (1 + reduction_factor * number_of_timesteps)
    """
    def __init__(self, param, initial, reduction_factor):
        super(self.__class__, self).__init__(param, initial, reduction_factor)
        self.epoch = 1
        if self.reduction_factor is None:
            self.reduction_factor = 0

    def decay(self):
        new_value = self.initial / (1 + self.reduction_factor*self.epoch)
        self.param.set_value(cast32(new_value))
        self.epoch += 1

    def simulate(self, initial, reduction_factor, epoch):
        new_value = initial / (1 + reduction_factor*epoch)
        return new_value


##### keep all the decay function subclasses above this line, and add them to the dictionary below! #######
_functions = {
    'linear': Linear,
    'exponential': Exponential,
    'montreal': Montreal
}

def get_decay_function(name, parameter, initial, reduction_factor):
    """
        This helper method returns the appropriate decay function given a string name. It looks up the appropriate function from the
        internal _functions dictionary.

        :param name: String representation of the decay function you want (normally grabbed from a config file)
        :type name: String

        :param parameter: the shared theano variable to use as the parameter to decay
        :type parameter: shared variable

        :param initial: String representation of the decay function you want (normally grabbed from a config file)
        :type initial: String

        :param reduction_factor: String representation of the decay function you want (normally grabbed from a config file)
        :type reduction_factor: String

        :return: The appropriate cost function, or raise NotImplementedError if it isn't found.
        :rtype: Method

        :raises: NotImplementedError
        """
    # standardize the input to be lowercase
    name = name.lower()
    # grab the appropriate activation function from the dictionary of decay functions
    func = _functions.get(name)
    # if it couldn't find the function (key didn't exist), raise a NotImplementedError
    if func is None:
        log.critical("Did not recognize decay function %s! Please use one of: ", str(name), str(_functions.keys()))
        raise NotImplementedError("Did not recognize decay function {0!s}! Please use one of: {1!s}".format(name, _functions.keys()))
    # return the found function
    return func(parameter, initial, reduction_factor)