"""
This module sets up plotting for values during training/testing.

It uses bokeh-server to create a local endpoint for serving graphs of data in the browser.

Adapted from Blocks: https://github.com/bartvm/blocks/blob/master/blocks/extensions/plot.py

Attributes
----------
BOKEH_AVAILABLE : bool
    Whether or not the user has Bokeh installed (calculated when it tries to import bokeh).
"""

__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger", "Blocks"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
import signal
import time
from subprocess import Popen, PIPE
import warnings
# third party libraries
try:
    from bokeh.plotting import (curdoc, cursession, figure, output_server, push, show)
    from bokeh.models.renderers import GlyphRenderer
    logging.getLogger("bokeh").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.INFO)
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False
    warnings.warn("Bokeh is not available - plotting is disabled. Please pip install bokeh to use Plot.")
# internal imports
from opendeep.monitor.monitor import MonitorsChannel, Monitor
from opendeep.monitor.monitor import COLLAPSE_SEPARATOR, TRAIN_MARKER, VALID_MARKER, TEST_MARKER
from opendeep.utils.misc import raise_to_list
from opendeep.optimization.optimizer import TRAIN_COST_KEY


log = logging.getLogger(__name__)


class Plot(object):
    """
    Live plotting of monitoring channels.

    .. warning::

      Depending on the number of plots, this can add 0.1 to 2 seconds per epoch
      to your training!

    In most cases it is preferable to start the Bokeh plotting server
    manually, so that your plots are stored permanently.

    To start the server manually, type ``bokeh-server`` in the command line.
    This will default to http://localhost:5006.
    If you want to make sure that you can access your plots
    across a network (or the internet), you can listen on all IP addresses
    using ``bokeh-server --ip 0.0.0.0``.

    Alternatively, you can set the ``start_server_flag`` argument to ``True``,
    to automatically start a server when training starts.
    However, in that case your plots will be deleted when you shut
    down the plotting server!

    .. warning::

       When starting the server automatically using the ``start_server_flag``
       argument, the extension won't attempt to shut down the server at the
       end of training (to make sure that you do not lose your plots the
       moment training completes). You have to shut it down manually (the
       PID will be shown in the logs). If you don't do this, this extension
       will crash when you try and train another model with
       ``start_server_flag`` set to ``True``, because it can't run two servers
       at the same time.

    """
    # Tableau 10 colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def __init__(self, bokeh_doc_name, monitor_channels=None, open_browser=False,
                 start_server=False, server_url='http://localhost:5006/',
                 colors=colors):
        """
        Initialize a Bokeh plot!

        Parameters
        ----------
        bokeh_doc_name : str
            The name of the Bokeh document. Use a different name for each
            experiment if you are storing your plots.
        channels : list(MonitorsChannel or Monitor)
            The monitor channels and monitors that you want to plot. The
            monitors within a :class:`MonitorsChannel` will be plotted together in a single
            figure.
        open_browser : bool, optional
            Whether to try and open the plotting server in a browser window.
            Defaults to ``True``. Should probably be set to ``False`` when
            running experiments non-locally (e.g. on a cluster or through SSH).
        start_server_flag : bool, optional
            Whether to try and start the Bokeh plotting server. Defaults to
            ``False``. The server started is not persistent i.e. after shutting
            it down you will lose your plots. If you want to store your plots,
            start the server manually using the ``bokeh-server`` command. Also
            see the warning above.
        server_url : str, optional
            Url of the bokeh-server. Ex: when starting the bokeh-server with
            ``bokeh-server --ip 0.0.0.0`` at ``alice``, server_url should be
            ``http://alice:5006``. When not specified the default configured
            to ``http://localhost:5006/``.
        colors : list(str)
            The list of string hex codes for colors to cycle through when creating new lines on the same figure.
        """
        # Make sure Bokeh is available
        if BOKEH_AVAILABLE:
            monitor_channels = raise_to_list(monitor_channels)
            if monitor_channels is None:
                monitor_channels = []

            self.channels = monitor_channels
            self.plots = {}
            self.colors = colors
            self.bokeh_doc_name = bokeh_doc_name
            self.server_url = server_url
            self.start_server(start_server_flag=start_server)

            # Create figures for each group of channels
            self.figures = []
            self.figure_indices = {}
            self.figure_color_indices = []

            # add a potential plot for train_cost
            self.figures.append(figure(title='{} #{}'.format(bokeh_doc_name, TRAIN_COST_KEY),
                                       logo=None,
                                       toolbar_location='right'))
            self.figure_color_indices.append(0)
            self.figure_indices[TRAIN_COST_KEY] = 0

            for i, channel in enumerate(self.channels):
                idx = i+1  # offset by 1 because of the train_cost figure
                assert isinstance(channel, MonitorsChannel) or isinstance(channel, Monitor), \
                    "Need channels to be type MonitorsChannel or Monitor. Found %s" % str(type(channel))
                # create the figure
                self.figures.append(figure(title='{} #{}'.format(bokeh_doc_name, channel.name),
                                           logo=None,
                                           toolbar_location='right'))
                self.figure_color_indices.append(0)
                # for each monitor in this channel, assign this figure to the monitor (and train/valid/test variants)
                if isinstance(channel, MonitorsChannel):
                    for monitor in channel.monitors:
                        self.figure_indices[COLLAPSE_SEPARATOR.join([channel.name, monitor.name])] = idx
                        if monitor.train_flag:
                            self.figure_indices[
                                COLLAPSE_SEPARATOR.join([channel.name, monitor.name, TRAIN_MARKER])
                            ] = idx
                        if monitor.valid_flag:
                            self.figure_indices[
                                COLLAPSE_SEPARATOR.join([channel.name, monitor.name, VALID_MARKER])
                            ] = idx
                        if monitor.test_flag:
                            self.figure_indices[
                                COLLAPSE_SEPARATOR.join([channel.name, monitor.name, TEST_MARKER])
                            ] = idx
                else:
                    self.figure_indices[channel.name] = idx
                    if channel.train_flag:
                        self.figure_indices[
                            COLLAPSE_SEPARATOR.join([channel.name, TRAIN_MARKER])
                        ] = idx
                    if channel.valid_flag:
                        self.figure_indices[
                            COLLAPSE_SEPARATOR.join([channel.name, VALID_MARKER])
                        ] = idx
                    if channel.test_flag:
                        self.figure_indices[
                            COLLAPSE_SEPARATOR.join([channel.name, TEST_MARKER])
                        ] = idx

            log.debug("Figure indices for monitors: %s" % str(self.figure_indices))

            if open_browser:
                show(self.figures)

    def update_plots(self, epoch, monitors):
        """
        Given the calculated monitors (collapsed name and value tuple), add its datapoint to the appropriate figure
        and update the figure in bokeh-server.

        Parameters
        ----------
        epoch : int
            The epoch (x-axis value in the figure).
        monitors : dict
            The dictionary of monitors calculated at this epoch. The dictionary is of the form
            {collapsed_monitor_name: value}. The name is the same that was used in the creation of the
            figures in the plot, so it is used as the key to finding the appropriate figure to add the
            data.
        """
        if BOKEH_AVAILABLE:
            for key, value in monitors.items():
                if key in self.figure_indices:
                    if key not in self.plots:
                        # grab the correct figure by its index for the key (same with the color)
                        fig = self.figures[self.figure_indices[key]]
                        color_idx = self.figure_color_indices[self.figure_indices[key]]
                        # split the channel from the monitor name
                        name = key.split(COLLAPSE_SEPARATOR, 1)
                        if len(name) > 1:
                            name = name[1]
                        else:
                            name = name[0]
                        # create a new line
                        fig.line([epoch], [value], legend=name,
                                 x_axis_label='iterations',
                                 y_axis_label='value', name=name,
                                 line_color=self.colors[color_idx % len(self.colors)])
                        color_idx += 1
                        # set the color index back in the figure list
                        self.figure_color_indices[self.figure_indices[key]] = color_idx
                        # grab the render object and put it in the plots dictionary
                        renderer = fig.select(dict(name=name))
                        self.plots[key] = renderer[0].data_source
                    else:
                        self.plots[key].data['x'].append(epoch)
                        self.plots[key].data['y'].append(value)
                        cursession().store_objects(self.plots[key])
            push()

    def start_server(self, start_server_flag):
        """
        Starts a bokeh-server at the default URL location.

        Parameters
        ----------
        start_server_flag : bool
            Whether to start the bokeh server.
        """
        if BOKEH_AVAILABLE:
            if start_server_flag:
                def preexec_fn():
                    """Prevents the server from dying on training interrupt."""
                    signal.signal(signal.SIGINT, signal.SIG_IGN)
                # Only memory works with subprocess, need to wait for it to start
                log.info('Starting plotting server on %s', self.server_url)
                self.sub = Popen('bokeh-server --ip 0.0.0.0 '
                                 '--backend memory'.split(),
                                 stdout=PIPE, stderr=PIPE, preexec_fn=preexec_fn)
                time.sleep(2)
                log.info('Plotting server PID: {}'.format(self.sub.pid))
            else:
                self.sub = None
            output_server(self.bokeh_doc_name, url=self.server_url)