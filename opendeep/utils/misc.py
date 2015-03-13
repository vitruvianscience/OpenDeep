"""
.. module:: misc

This module contains utils that are general and can't be grouped logically into the other opendeep.utils modules.
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
import theano
import theano.tensor as T
# internal imports
from opendeep import trunc, safe_zip

log = logging.getLogger(__name__)

def make_time_units_string(time):
    """
    This takes a time (in seconds) and converts it to an easy-to-read format with the appropriate units.

    :param time: the time to make into a string (in seconds)
    :type time: Integer

    :return: an easy-to-read string representation of the time
    :rtype: String
    """
    # Show the time with appropriate units.
    if time < 1:
        return trunc(time*1000)+" milliseconds"
    elif time < 60:
        return trunc(time)+" seconds"
    elif time < 3600:
        return trunc(time/60)+" minutes"
    else:
        return trunc(time/3600)+" hours"
    
def raise_to_list(input):
    """
    This will take an input and raise it to a List (if applicable)

    :param input: object to raise to a list
    :type input: Object

    :return: the object as a list, or none
    :rtype: List or None
    """
    if input is None:
        return None
    elif isinstance(input, list):
        return input
    else:
        return [input]
    
def stack_and_shared(_input):
    """
    This will take a list of input variables, turn them into theano shared variables, and return them stacked in a single tensor.

    :param _input: list of input variables
    :type _input: list, object, or none

    :return: symbolic tensor of the input variables stacked, or none
    :rtype: Tensor or None
    """
    if _input is None:
        return None
    elif isinstance(_input, list):
        shared_ins = []
        for _in in _input:
            try:
                shared_ins.append(theano.shared(_in))
            except TypeError as _:
                shared_ins.append(_in)
        return T.stack(shared_ins)
    else:
        try:
            _output = [theano.shared(_input)]
        except TypeError as _:
            _output = [_input]
        return T.stack(_output)
    
def concatenate_list(input, axis=0):
    """
    This takes a list of tensors and concatenates them along the axis specified (0 by default)

    :param input: list of tensors
    :type input: List

    :param axis: axis to concatenate along
    :type axis: Integer

    :return: the concatenated tensor, or None
    :rtype: Tensor or None
    """
    if input is None:
        return None
    elif isinstance(input, list):
        return T.concatenate(input, axis=axis)
    else:
        return input
    
    
def closest_to_square_factors(n):
    """
    This function finds the integer factors that are closest to the square root of a number. (Useful for finding the closest
    width/height of an image you want to make square)

    :param n: The number to find its closest-to-square root factors.
    :type n: Integer

    :return: the tuple of (factor1, factor2) that are closest to the square root
    :rtype: Tuple
    """
    test = numpy.ceil(numpy.sqrt(float(n)))
    while not (n/test).is_integer():
        test-=1
    if test < 1:
        test = 1
    return int(test), int(n/test)

def get_shared_values(variables, borrow=False):
    """
    This will return the values from a list of shared variables.

    :param variables: the list of shared variables to grab values
    :type variables: List(shared_variable)

    :param borrow: the borrow argument for theano shared variable's get_value() method
    :type borrow: Boolean

    :return: the list of values held by the shared variables
    :rtype: List
    """
    try:
        values = [variable.get_value(borrow=borrow) for variable in variables]
    except AttributeError as e:
        log.exception("Cannot get values, there was an AttributeError %s",
                      str(e))
        raise
    return values

def set_shared_values(variables, values, borrow=False):
    """
    This sets the shared variables' values from a list of variables to the values specified in a list

    :param variables: the list of shared variables to set values
    :type variables: List(shared_variable)

    :param values: the list of values to set the shared variables to
    :type values: List

    :param borrow: the borrow argument for theano shared variable's set_value() method
    :type borrow: Boolean

    :raises: ValueError if the list of variables and the list of values are different lengths, AttributeError if no .set_value() function
    """
    # use the safe_zip wrapper to ensure the variables and values lists are of the same length
    for variable, value in safe_zip(variables, values):
        try:
            variable.set_value(value, borrow=borrow)
        except AttributeError as e:
            log.exception("Cannot set values, there was an AttributeError %s",
                          str(e))
            raise