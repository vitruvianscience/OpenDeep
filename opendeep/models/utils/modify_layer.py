"""
This module defines the generic ModifyLayer class -
which doesn't have learnable parameters but takes inputs and modifies
them to outputs.
"""
# standard libraries
import logging
import os
import time
# third party libraries
import theano
import theano.tensor as T
from theano.compat.python2x import OrderedDict  # use this compatibility OrderedDict
# internal references
from opendeep.utils.decorators import init_optimizer
from opendeep.utils import file_ops
from opendeep.utils.constructors import function
from opendeep.utils.misc import set_shared_values, get_shared_values, \
    make_time_units_string, raise_to_list, add_kwargs_to_dict
from opendeep.utils.file_ops import mkdir_p

try:
    import cPickle as pickle
except ImportError:
    import pickle

log = logging.getLogger(__name__)


class ModifyLayer(object):
    """
    The :class:`ModifyLayer` is a generic class for a neural net layer that doesn't have
    learnable parameters. This includes things like batch normalization and dropout.
    """

    def __init__(self, inputs_hook=None,
                 input_size=None, output_size=None,
                 outdir=None,
                 **kwargs):
        raise NotImplementedError("ModifyLayer is in development.")
