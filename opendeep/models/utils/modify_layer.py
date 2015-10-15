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
    def __init__(self, inputs=None, **kwargs):
        self._classname = self.__class__.__name__
        self.inputs = raise_to_list(inputs)
        self.args = {}
        self.args = add_kwargs_to_dict(kwargs.copy(), self.args)
        self.args['inputs'] = self.inputs

    def get_inputs(self):
        """
        This should return the input(s) to the model's computation graph as a list. This only includes inputs for the
        run function, not any inputs used for supervised training.

        .. note::
            This should normally return the same theano variable list that is used in the inputs= argument to the
            f_run function when running the Model on an input.

        Returns
        -------
        Theano variable or List(theano variable)
            Theano variables representing the input(s) to the model's 'run' computation.

        Raises
        ------
        NotImplementedError
            If the function hasn't been implemented for the specific model.
        """
        log.critical("%s does not have a get_inputs function!", self._classname)
        raise NotImplementedError("Please implement a get_inputs method for %s" % self._classname)

    def get_outputs(self):
        """
        This method will return the model's output variable expression from the computational graph.
        This should be what is given for the outputs= part of the 'f_run' function from `self.run()`.

        This will be used for creating hooks to link models together,
        where these outputs can be strung as the inputs or hiddens to another model :)

        Returns
        -------
        theano expression or list(theano expression)
            Theano expression(s) of the outputs from this model's computation graph.

        Examples
        --------
        Here is an example showing the `get_outputs()` method in the GSN model used in an `inputs` hook
        to a SoftmaxLayer model::

            from opendeep.models import GSN, SoftmaxLayer
            gsn = GSN(inputs=28*28, hiddens=1000, layers=2, walkbacks=4)
            softmax = SoftmaxLayer(inputs=zip(gsn.output_size, gsn.get_outputs()), output_size=10)

        Raises
        ------
        NotImplementedError
            If the function hasn't been implemented for the specific model.
        """
        log.critical("%s get_outputs method not implemented!", self._classname)
        raise NotImplementedError("Please implement a get_outputs method for %s" % self._classname)

    def get_updates(self):
        """
        This should return any theano updates from the model (used for things like random number generators).
        Most often comes from theano's 'scan' op. Check out its documentation at
        http://deeplearning.net/software/theano/library/scan.html.

        This is used with the :class:`Optimizer` to create the training function - the 'updates='
        part of the theano function.

        Returns
        -------
        iterable over pairs (shared_variable, new_expression)
            Updates from the theano computation for the model to be used during Optimizer.train()
            (but not including training parameter updates - those are calculated by the :class:`Optimizer`)
            These are expressions for new SharedVariable values.
        """
        # TODO: should we do the parameter decays from get_decay_params() in the model updates?
        # TODO: Right now I'm not because it seems less modular
        # by default, assume the model doesn't have updates - it's your job to return them in this method.
        return None

    def get_decay_params(self):
        """
        If the layer requires any of its internal parameters to decay over time during training, return the list
        of the :class:`DecayFunction` (from opendeep.utils.decay) objects here so the :class:`Optimizer` can decay
        them each epoch.

        Most layers don't need to decay parameters, so we return an empty list by default. Please override this method
        if you need to decay some variables.

        Returns
        -------
        list(:class:`DecayFunction`)
            List of opendeep.utils.decay_functions.DecayFunction objects of the parameters to decay for this model.
            Defaults to an empty list - no decay parameters.
        """
        # no decay parameters by default
        return []

    def get_switches(self):
        """
        This method returns a list of shared theano variables representing switches for values in the model that
        get turned on or off for training/testing.
        The variables should be set to either 0. or 1.
        These switch variables are used in theano Switch operations, such as adding noise during training and removing
        it during testing.

        Returns
        -------
        list
            List of SharedVariable used to set the Switches. Defaults to an empty list.
        """
        return []

    def flip_switches(self):
        """
        This helper method flips all Theano switches specified by `get_switches()` to 0. or 1. (the opposite value
        that the switch is currently set to).
        """
        switches = raise_to_list(self.get_switches())
        log.debug("Flipping %d switches for %s!" % (len(switches), self._classname))
        [switch.set_value(1. - switch.get_value()) for switch in switches]
        if self.switches_on is not None:
            self.switches_on = not self.switches_on

    def turn_off_switches(self):
        """
        This helper method turns all Theano switches by `get_switches()` to their off position of 0./False
        """
        switches = raise_to_list(self.get_switches())
        log.debug("Turning off %d switches for %s!" % (len(switches), self._classname))
        [switch.set_value(0.) for switch in switches]
        self.switches_on = False

    def turn_on_switches(self):
        """
        This helper method turns all Theano switches by `get_switches()` to their on position of 1./True
        """
        switches = raise_to_list(self.get_switches())
        log.debug("Turning on %d switches for %s!" % (len(switches), self._classname))
        [switch.set_value(1.) for switch in switches]
        self.switches_on = True

    def set_switches(self, values):
        """
        This helper method sets all Theano switches from `get_switches()` to the `values` parameter specified.

        Parameters
        ----------
        values : list(boolean)
        """
        switches = raise_to_list(self.get_switches())
        values = raise_to_list(values)
        values = [1. if val else 0. for val in values]
        assert len(switches) == len(values), "Switches (len %d) needs to be same length as values (len %d)!" % \
                                             (len(switches), len(values))
        log.debug("Setting specified values for %d switches!" % len(switches))
        [switch.set_value(val) for switch, val in zip(switches, values)]
        self.switches_on = None