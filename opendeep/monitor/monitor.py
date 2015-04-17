"""
.. module:: monitor

This module sets up a Monitor object for keeping track of some values during training/testing
"""

__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
# third party
from theano.compat.six import string_types  # for basestring compatability
# internal
from opendeep.utils.misc import raise_to_list

log = logging.getLogger(__name__)

COLLAPSE_SEPARATOR = "/"
TRAIN_MARKER = 'train'
VALID_MARKER = 'valid'
TEST_MARKER = 'test'

class MonitorsChannel(object):
    """
    A MonitorsChannel is a list of monitors that belong together
    """
    def __init__(self, name, monitors=None):
        """
        Initializes a channel with name and a potential starting list of monitors to include.

        Names of channels have to be unique from each other.
        """
        monitors = raise_to_list(monitors)
        if monitors is not None:
            # make sure the input monitors are actually Monitors
            for monitor in monitors:
                assert isinstance(monitor, Monitor), \
                    "Input monitors need to all be type Monitor. Found %s" % str(type(monitor))
            # initialize the list with these monitors
            self.monitors = monitors
        else:
            # otherwise, start empty
            self.monitors = []
        # make sure the channel name is a string
        assert isinstance(name, string_types), "name needs to be a string. found %s" % str(type(name))
        self.name = name

    def add(self, monitor):
        """
        Adds a monitor (or list of monitors) to the channel
        """
        monitors = raise_to_list(monitor)
        # make sure the input monitors are actually Monitors
        for monitor in monitors:
            assert isinstance(monitor, Monitor), \
                "Input monitors need to all be type Monitor. Found %s" % str(type(monitor))
            # check if monitor is already in the channel - if it is, skip.
            if monitor.expression in self.get_monitor_expressions():
                monitors.remove(monitor)
            else:
                names = self.get_monitor_names()
                # check if the monitor has the same name as one in the channel; if so, rename it with a number added.
                # for example, if a monitor with name 'a' already exists in the channel, it will be renamed to 'a_0'.
                if monitor.name in names:
                    i = 0
                    potential_name = '_'.join([monitor.name, str(i)])
                    while potential_name in names or i > 10000:
                        i += 1
                        potential_name = '_'.join([monitor.name, str(i)])
                    # found the next open name, so rename the monitor.
                    log.info("Renaming monitor %s (from Channel %s) to %s.", monitor.name, self.name, potential_name)
                    monitor.name = potential_name
                # add the monitor to the list!
                self.monitors.append(monitor)

    def pop(self, name):
        """
        returns a monitor with the name and removes it from the list
        """
        for monitor in self.monitors:
            if monitor.name.lower() == name.lower():
                self.monitors.remove(monitor)
                return monitor
        log.error("Couldn't find monitor %s in channel %s.", name, self.name)
        return None

    def remove(self, name):
        """
        removes a monitor with the name from the list
        """
        for monitor in self.monitors:
            if monitor.name.lower() == name.lower():
                self.monitors.remove(monitor)
        log.error("Couldn't find monitor %s in channel %s.", name, self.name)

    def get_monitor_names(self):
        return [monitor.name for monitor in self.monitors]

    def get_monitor_expressions(self):
        return [monitor.expression for monitor in self.monitors]

    def get_train_monitors(self):
        return [monitor for monitor in self.monitors if monitor.train_flag]

    def get_valid_monitors(self):
        return [monitor for monitor in self.monitors if monitor.valid_flag]

    def get_test_monitors(self):
        return [monitor for monitor in self.monitors if monitor.test_flag]

    def get_monitors(self, train=None, valid=None, test=None):
        # if all the flags not given, just return all the monitors
        if train is None and valid is None and test is None:
            return self.monitors
        else:
            # otherwise, add the monitors with the appropriate flags, make sure not to duplicate but keep order
            # as train - valid - test
            monitors = []
            if train:
                for monitor in self.get_train_monitors():
                    if monitor not in monitors:
                        monitors.append(monitor)
            if valid:
                for monitor in self.get_valid_monitors():
                    if monitor not in monitors:
                        monitors.append(monitor)
            if test:
                for monitor in self.get_test_monitors():
                    if monitor not in monitors:
                        monitors.append(monitor)
            return monitors


class Monitor(object):
    """
    A Monitor to make managing values to output during training/testing easy.
    """
    def __init__(self, name, expression, train=True, valid=False, test=False):
        """
        This initializes a Monitor representation.

        :param name: a string of what to call the monitor
        :type name: str

        :param expression:
        """
        # make sure the monitor name is a string
        assert isinstance(name, string_types), "name needs to be a string. found %s" % str(type(name))
        self.name = name
        self.expression = expression
        self.train_flag = train
        self.valid_flag = valid
        self.test_flag  = test


def collapse_channels(monitor_channels, train=None, valid=None, test=None):
    """
    This function takes a list of monitor channels and collapses them into a list of tuples (collapsed_name, expression)

    :param monitor_channels: list of MonitorsChannels to collapse
    :type monitor_channels: MonitorsChannel
    :return: list of tuples
    :rtype: list of tuples
    """
    monitor_channels = raise_to_list(monitor_channels)
    collapsed = []
    for channel in monitor_channels:
        # make sure it is the right type
        assert isinstance(channel, MonitorsChannel), "Need input monitor_channels to be MonitorsChannel! Found %s" % \
            str(type(channel))
        # grab the channel's monitors
        monitors = channel.monitors
        # if flags are all None, just grab one copy of all the monitors.
        if train is None and valid is None and test is None:
            # collapse their names with the channel name
            names = [COLLAPSE_SEPARATOR.join([channel.name, monitor.name]) for monitor in monitors]
            expressions = [monitor.expression for monitor in monitors]

        else:
            # collapse their names with the channel name and the train/valid/test ending
            names = []
            expressions = []
            for monitor in monitors:
                if monitor.train_flag and train:
                    names.append(COLLAPSE_SEPARATOR.join([channel.name, monitor.name, TRAIN_MARKER]))
                    expressions.append(monitor.expression)
                if monitor.valid_flag and valid:
                    names.append(COLLAPSE_SEPARATOR.join([channel.name, monitor.name, VALID_MARKER]))
                    expressions.append(monitor.expression)
                if monitor.test_flag and test:
                    names.append(COLLAPSE_SEPARATOR.join([channel.name, monitor.name, TEST_MARKER]))
                    expressions.append(monitor.expression)

        # extend the list of tuples
        collapsed.extend(zip(names, expressions))

    return collapsed

def values_dict_from_collapsed(names, vals):
    """
    This function takes two lists (in the same order) of collapsed MonitorsChannel and Monitor names from the function
    above, and the values computed for these monitors by some function, and returns a dictionary of dictionaries
    representing the same structure as MonitorsChannel.

    :param names: list of names that were computed from collapsing MonitorsChannels.
    :type names: list of string
    :param vals: list of values computed corresponding to the names
    :type vals: list of floats
    :return: dictionary of dictionaries where the top level is the MonitorsChannel, and the lower level are the Monitors
    :rtype: dict of dict
    """
    names = raise_to_list(names)
    vals = raise_to_list(vals)
    # make sure the lists have the same length
    assert len(names) == len(vals), "Names and values need to be same length. Found %d, %d" % (len(names), len(vals))

    reconstructed = {}
    for name, val in zip(names, vals):
        channel_name, monitor_name = name.split(COLLAPSE_SEPARATOR, 1)
        if channel_name in reconstructed:
            reconstructed[channel_name].update(monitor_name, val)
        else:
            reconstructed[channel_name] = {monitor_name: val}