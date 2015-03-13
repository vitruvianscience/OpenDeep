'''
Code used throughout the entire OpenDeep package.

Lambda functions taken from Li Yao's GSN code:
https://github.com/yaoli/GSN/blob/master/model.py

Others taken from Pylearn2:
https://github.com/lisa-lab/pylearn2
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# third-party libraries
import theano
import numpy

cast32      = lambda x: numpy.cast['float32'](x)
cast_floatX = lambda x: numpy.cast[theano.config.floatX](x)
trunc       = lambda x: str(x)[:8]
logit       = lambda p: numpy.log(p / (1 - p))
binarize    = lambda x: cast32(x >= 0.5)
sigmoid     = lambda x: cast32(1. / (1 + numpy.exp(-x)))

def function(*args, **kwargs):
    """
    A wrapper around theano.function that disables the on_unused_input error.
    Almost no part of OpenDeep can assume that an unused input is an error, so
    the default from theano is inappropriate for this project.
    """
    return theano.function(*args, on_unused_input='warn', **kwargs)

def grad(*args, **kwargs):
    """
    A wrapper around theano.gradient.grad that disable the disconnected_inputs
    error. Almost no part of OpenDeep can assume that a disconnected input
    is an error.
    """
    return theano.gradient.grad(*args, disconnected_inputs='warn', **kwargs)

def sharedX(value, name=None, borrow=False, dtype=None):
    """
    Transform value into a theano shared variable of type floatX
    """
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(theano._asarray(value, dtype=dtype),
                         name=name,
                         borrow=borrow)

def as_floatX(variable):
    """
    Casts a given variable into dtype `config.floatX`. Numpy ndarrays will
    remain numpy ndarrays, python floats will become 0-D ndarrays and
    all other types will be treated as theano tensors.
    """
    if isinstance(variable, float):
        return numpy.cast[theano.config.floatX](variable)

    if isinstance(variable, numpy.ndarray):
        return numpy.cast[theano.config.floatX](variable)

    return theano.tensor.cast(variable, theano.config.floatX)


def constantX(value):
    """
    Returns a constant of value `value` with floatX dtype.
    """
    return theano.tensor.constant(
        numpy.asarray(value, dtype=theano.config.floatX)
    )

def safe_zip(*args):
    """
    Like zip, but ensures arguments are of same length
    """
    base = len(args[0])
    for i, arg in enumerate(args[1:]):
        if len(arg) != base:
            raise ValueError("Argument 0 has length %d but argument %d has "
                             "length %d" % (base, i+1, len(arg)))
    return zip(*args)

def make_shared_variables(variable_list, borrow=True):
    """
    Takes a list of variables to make into theano shared variables of type floatX.
    """
    # Borrow is true by default
    return (sharedX(variable, borrow=borrow) if variable is not None else None for variable in variable_list)