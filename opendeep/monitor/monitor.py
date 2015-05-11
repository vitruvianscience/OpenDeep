"""
This module sets up monitors for keeping track of some values during training/testing.

In essence, it is a wrapper around shared variables that allows you to output their values to different
locations, such as files, databases, or plots.

Attributes
----------
COLLAPSE_SEPARATOR : str
    The string separator to use when collapsing monitor names from monitor channels.
TRAIN_MARKER : str
    The string indicator to append to collapsed monitor names to indicate they should be used on training data.
VALID_MARKER : str
    The string indicator to append to collapsed monitor names to indicate they should be used on validation data.
TEST_MARKER : str
    The string indicator to append to collapsed monitor names to indicate they should be used on testing data.
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
    A :class:`MonitorsChannel` is a list of monitors that logically belong together. For example, the means of model
    weight matrices.

    Attributes
    ----------
    name : str
        The channel's unique name.
    monitors : list
        The list of :class:`Monitor` in this channel.
    """
    def __init__(self, name, monitors=None):
        """
        Initializes a channel with `name` and a potential starting list of :class:`Monitor` to include.

        Names of channels have to be unique from each other.

        Parameters
        ----------
        name : str
            The unique name to give this channel.
        monitors : Monitor, list(Monitor), optional
            The starting monitor(s) to use for this channel.
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
        Adds a :class:`Monitor` (or list of monitors) to the channel.

        This will append `monitor` to `self.monitors`.

        Parameters
        ----------
        monitor : Monitor or list(Monitor)
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
        Returns a :class:`Monitor` with the name `name` and removes it from the `self.monitors` list.

        Parameters
        ----------
        name : str
            Name of the monitor to pop from the channel.

        Returns
        -------
        Monitor
            The monitor with `name`, otherwise None.
        """
        for monitor in self.monitors:
            if monitor.name.lower() == name.lower():
                self.monitors.remove(monitor)
                return monitor
        log.error("Couldn't find monitor %s in channel %s.", name, self.name)
        return None

    def remove(self, name):
        """
        Removes a :class:`Monitor` with the name `name` from the `self.monitors` list.

        Parameters
        ----------
        name : str
            Name of the monitor to remove from the channel.
        """
        for monitor in self.monitors:
            if monitor.name.lower() == name.lower():
                self.monitors.remove(monitor)
        log.error("Couldn't find monitor %s in channel %s.", name, self.name)

    def get_monitor_names(self):
        """
        Returns the list of names of the Monitors in this channel.

        Returns
        -------
        list
            List of string names for the monitors in this channel.
        """
        return [monitor.name for monitor in self.monitors]

    def get_monitor_expressions(self):
        """
        Returns the list of computation expressions (or variable) of the Monitors in this channel.

        Returns
        -------
        list
            List of theano expressions for the monitors in this channel.
        """
        return [monitor.expression for monitor in self.monitors]

    def get_train_monitors(self):
        """
        Returns the list of monitors with `train_flag` set to True in this channel. (The monitors to be run
        on training data).

        Returns
        -------
        list
            List of monitors with `train_flag=True`.
        """
        return [monitor for monitor in self.monitors if monitor.train_flag]

    def get_valid_monitors(self):
        """
        Returns the list of monitors with `valid_flag` set to True in this channel. (The monitors to be run
        on validation data).

        Returns
        -------
        list
            List of monitors with `valid_flag=True`.
        """
        return [monitor for monitor in self.monitors if monitor.valid_flag]

    def get_test_monitors(self):
        """
        Returns the list of monitors with `test_flag` set to True in this channel. (The monitors to be run
        on testing data).

        Returns
        -------
        list
            List of monitors with `test_flag=True`.
        """
        return [monitor for monitor in self.monitors if monitor.test_flag]

    def get_monitors(self, train=None, valid=None, test=None):
        """
        Returns the list of monitors with the given flags. If all input flags set to None (left at their default
        values), this will just return the full `self.monitors` list. For multiple flags, this returns the
        union of the sets of monitors with each flag.

        Parameters
        ----------
        train : bool, optional
            Whether to return monitors with `train_flag=True`.
        valid : bool, optional
            Whether to return monitors with `valid_flag=True`.
        test : bool, optional
            Whether to return monitors with `test_flag=True`.

        Returns
        -------
        list
            List of monitors containing the appropriate flags.
        """
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
    A :class:`Monitor` is a way to make managing values to output during training/testing easy.

    It associates a friendly name with a variable or expression, what dataset(s) to evaluate the variable or expression,
    and where to output the result.

    Attributes
    ----------
    name : str
        Unique name to represent the monitor.
    expression : theano expression/variable.
        The computation for the value of the Monitor.
    train_flag : bool
        Whether to run this monitor on training data.
    valid_flag : bool
        Whether to run this monitor on validation data.
    test_flag : bool
        Whether to run this monitor on testing data.
    out_service : OutService
        The :class:`OutService` to for this monitor - where its output goes.
    """
    def __init__(self, name, expression, out_service=None, train=True, valid=False, test=False):
        """
        This initializes a :class:`Monitor` representation.

        Parameters
        ----------
        name : str
            Unique name to represent the monitor.
        expression : theano expression/variable.
            The computation for the value of the Monitor.
        out_service : OutService
            The :class:`OutService` to for this monitor - where its output goes.
        train_flag : bool
            Whether to run this monitor on training data.
        valid_flag : bool
            Whether to run this monitor on validation data.
        test_flag : bool
            Whether to run this monitor on testing data.
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
    This function takes a list of :class:`MonitorsChannel` and flattens them into a
    list of tuples (collapsed_name, expression, out_service).

    Names are collapsed according to the convention:
    `COLLAPSE_SEPARATOR`.join([channel_name, monitor_name]).append(train/valid/test marker)

    Parameters
    ----------
    monitor_channels : list(MonitorsChannel or Monitor)
        The list of MonitorsChannel or Monitor to collapse.
    train : bool, optional
        Whether to collapse the monitors to be used on training data.
    valid : bool, optional
        Whether to collapse the monitors to be used on validation data.
    test : bool, optional
        Whether to collapse the monitors to be used on testing data.

    Returns
    -------
    list
        List of (collapsed_monitor_name, monitor_expression, out_service) tuples.
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