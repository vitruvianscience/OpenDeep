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
import os
# third party
from theano.compat.six import string_types  # for basestring compatability
# internal
from opendeep.utils.misc import raise_to_list
from opendeep.monitor.out_service import FileService

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
    def __init__(self, name, expression, out_service=None, train=True, valid=False, test=False):
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
        self.out_service = out_service
        # remove redundant files made by the fileservice
        if isinstance(self.out_service, FileService):
            if not self.train_flag:
                os.remove(self.out_service.train_filename)
            if not self.valid_flag:
                os.remove(self.out_service.valid_filename)
            if not self.test_flag:
                os.remove(self.out_service.test_filename)

def collapse_channels(monitor_channels, train=None, valid=None, test=None):
    """
    This function takes a list of monitor channels and collapses them into a
    list of tuples (collapsed_name, expression, out_service)

    :param monitor_channels: list of MonitorsChannels or Monitors to collapse
    :type monitor_channels: list of MonitorsChannel or Monitor
    :return: list of tuples
    :rtype: list of tuples
    """
    monitor_channels = raise_to_list(monitor_channels)
    collapsed = []
    for channel in monitor_channels:
        # make sure it is the right type
        if isinstance(channel, MonitorsChannel):
            # grab the channel's monitors
            monitors = channel.monitors
            is_channel = True
        elif isinstance(channel, Monitor):
            # or if it is a monitor already, just return it as a single item list
            monitors = raise_to_list(channel)
            is_channel = False
        else:
            raise AssertionError("Expected Monitor or MonitorChannel, found %s" % str(type(channel)))

        # if flags are all None, just grab one copy of all the monitors.
        if train is None and valid is None and test is None:
            # collapse their names with the channel name
            if is_channel:
                names = [COLLAPSE_SEPARATOR.join([channel.name, monitor.name]) for monitor in monitors]
            else:
                names = [monitor.name for monitor in monitors]
            expressions = [monitor.expression for monitor in monitors]
            services = [monitor.out_service for monitor in monitors]

        else:
            # collapse their names with the channel name and the train/valid/test ending
            names = []
            expressions = []
            services = []
            for monitor in monitors:
                if monitor.train_flag and train:
                    if is_channel:
                        names.append(COLLAPSE_SEPARATOR.join([channel.name, monitor.name, TRAIN_MARKER]))
                    else:
                        names.append(COLLAPSE_SEPARATOR.join([monitor.name, TRAIN_MARKER]))
                    expressions.append(monitor.expression)
                    services.append(monitor.out_service)
                if monitor.valid_flag and valid:
                    if is_channel:
                        names.append(COLLAPSE_SEPARATOR.join([channel.name, monitor.name, VALID_MARKER]))
                    else:
                        names.append(COLLAPSE_SEPARATOR.join([monitor.name, VALID_MARKER]))
                    expressions.append(monitor.expression)
                    services.append(monitor.out_service)
                if monitor.test_flag and test:
                    if is_channel:
                        names.append(COLLAPSE_SEPARATOR.join([channel.name, monitor.name, TEST_MARKER]))
                    else:
                        names.append(COLLAPSE_SEPARATOR.join([monitor.name, TEST_MARKER]))
                    expressions.append(monitor.expression)
                    services.append(monitor.out_service)

        # extend the list of tuples
        collapsed.extend(zip(names, expressions, services))

    return collapsed