"""
Functions used for decaying Theano parameters as shared variables.
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
import numpy
import theano.compat.six as six
# internal references
from opendeep import as_floatX
from opendeep.utils.decorators import inherit_docs

log = logging.getLogger(__name__)

class DecayFunction(object):
    """
    Interface for a parameter decay function (like learning rate, noise levels, etc.)
    """
    def __init__(self, param, initial, reduction_factor):
        """
        A generic class for decaying a theano variable.

        Parameters
        ----------
        param : shared variable
            The theano variable you want to decay. This must already be a shared variable.
        initial : float
            The initial value the variable should have.
        reduction_factor : float
            The amount of reduction (depending on subclass's algorithm) each epoch.
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
        self.param.set_value(as_floatX(self.initial))
        self.reduction_factor = reduction_factor

    def decay(self):
        """
        This will decay the shared variable according to the decay rule of the subclass.
        """
        log.critical('Parameter decay function %s does not have a decay method!', str(type(self)))
        raise NotImplementedError()

    def reset(self):
        """
        Resets this shared variable to the initial value provided during the constructor.
        """
        self.param.set_value(self.initial)

    def simulate(self, initial, reduction_factor, epoch):
        """
        This will take an initial value for a hypothetical variable, the reduction factor appropriate to the
        subclass's decay function, and the number of decays (epoch) you want to see the simulated result after.

        Parameters
        ----------
        initial : float
            Initial value for the variable when simulating.
        reduction_factor : float
            The appropriate reduction factor parameter (for the subclass) when simulating.
        epoch : int
            Number of timesteps to simulate.

        Returns
        -------
        float
            The simulated value depending on the reduction after the given number of epochs.
        """
        log.critical('Parameter decay function %s does not have a simulate method!', str(type(self)))
        raise NotImplementedError()


@inherit_docs
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
        self.param.set_value(as_floatX(numpy.max([0, new_value])))

    def simulate(self, initial_value, reduction_factor, epoch):
        new_value = initial_value - reduction_factor*epoch
        return numpy.max([0, new_value])


@inherit_docs
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
        self.param.set_value(as_floatX(new_value))

    def simulate(self, initial_value, reduction_factor, epoch):
        new_value = initial_value*pow(reduction_factor, epoch)
        return new_value


@inherit_docs
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
        self.param.set_value(as_floatX(new_value))
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
    This helper method returns the appropriate decay function given a string name.
    It looks up the appropriate function from the internal _functions dictionary.

    Parameters
    ----------
    name : str
        String representation of the decay function you want.
    parameter : shared variable
        The shared theano variable to use as the parameter to decay.
    initial : float
        The initial value to set for the `parameter`.
    reduction_factor : float
        The amount of reduction (depending on subclass's algorithm) each epoch.

    Returns
    -------
    function
        The appropriate decay function (as an instantiated DecayFunction class).

    Raises
    ------
    NotImplementedError
        If the name can't be found in the _functions dictionary.
    """
    # make sure name is a string
    if isinstance(name, six.string_types):
        # standardize the input to be lowercase
        name = name.lower()
        # grab the appropriate activation function from the dictionary of decay functions
        func = _functions.get(name)
        # if it couldn't find the function (key didn't exist), raise a NotImplementedError
        if func is None:
            log.critical("Did not recognize decay function %s! Please use one of: ", str(name), str(_functions.keys()))
            raise NotImplementedError(
                "Did not recognize decay function {0!s}! Please use one of: {1!s}".format(name, _functions.keys())
            )
        # return the found function
        return func(parameter, initial, reduction_factor)
    # otherwise we don't know
    else:
        log.critical("Decay function not implemented for %s with type %s", str(name), str(type(name)))
        raise NotImplementedError("Decay function not implemented for %s with type %s", str(name), str(type(name)))
