"""
This module sets up plotting for values during training/testing.

It uses bokeh-server to create a local endpoint for serving graphs of data in the browser.

Attributes
----------
BOKEH_AVAILABLE : bool
    Whether or not the user has Bokeh installed (calculated when it tries to import bokeh).
"""
# standard libraries
import logging
import warnings
# third party libraries
try:
    from bokeh.client import push_session
    from bokeh.plotting import (curdoc, output_server, figure)
    from bokeh.models.renderers import GlyphRenderer
    logging.getLogger("bokeh").setLevel(logging.INFO)  # only log info and up priority for bokeh
    logging.getLogger("urllib3").setLevel(logging.INFO)  # only log info and up priority for urllib3
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False
    warnings.warn("Bokeh is not available - plotting is disabled. Please pip install bokeh to use Plot.")
# internal imports
from opendeep.monitor.monitor import MonitorsChannel, Monitor
from opendeep.monitor.monitor import COLLAPSE_SEPARATOR, TRAIN_MARKER, VALID_MARKER, TEST_MARKER
from opendeep.utils.misc import raise_to_list


log = logging.getLogger(__name__)


class Plot(object):
    """
    Live plotting of monitoring channels.

    .. warning::

      Depending on the number of plots, this can add ~0.1 to 2 seconds per epoch
      to your training!

    You must start the Bokeh plotting server
    manually, so that your plots are stored permanently.

    To start the server manually, type ``bokeh serve`` in the command line.
    This will default to http://localhost:5006.
    If you want to make sure that you can access your plots
    across a network (or the internet), you can listen on all IP addresses
    using ``bokeh serve --ip 0.0.0.0``.
    """
    # Tableau 10 colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def __init__(self, bokeh_doc_name, monitor_channels=None, open_browser=False,
                 server_url='http://localhost:5006/',
                 colors=colors):
        """
        Initialize a Bokeh plot!

        Parameters
        ----------
        bokeh_doc_name : str
            The name of the Bokeh document. Use a different name for each
            experiment if you are storing your plots.
        monitor_channels : list(MonitorsChannel or Monitor)
            The monitor channels and monitors that you want to plot. The
            monitors within a :class:`MonitorsChannel` will be plotted together in a single
            figure.
        open_browser : bool, optional
            Whether to try and open the plotting server in a browser window.
            Defaults to ``True``. Should probably be set to ``False`` when
            running experiments non-locally (e.g. on a cluster or through SSH).
        server_url : str, optional
            Url of the bokeh server. Ex: when starting the bokeh server with
            ``bokeh serve --ip 0.0.0.0`` at ``alice``, server_url should be
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
            self.colors = colors
            self.bokeh_doc_name = bokeh_doc_name
            self.server_url = server_url

            session = push_session(curdoc(), session_id=self.bokeh_doc_name, url=self.server_url)

            # Create figures for each group of channels
            self.data_sources = {}
            self.figures = []
            self.figure_indices = {}
            self.line_color_idx = 0

            for i, channel in enumerate(self.channels):
                idx = i
                assert isinstance(channel, MonitorsChannel) or isinstance(channel, Monitor), \
                    "Need channels to be type MonitorsChannel or Monitor. Found %s" % str(type(channel))

                # create the figure for this channel
                fig = figure(title='{} #{}'.format(bokeh_doc_name, channel.name),
                             x_axis_label='epochs',
                             y_axis_label='value',
                             logo=None,
                             toolbar_location='right')
                # keep track of the line colors so we can rotate them around in the same manner across figures
                self.line_color_idx = 0

                # for each monitor in this channel, create the line (and train/valid/test variants if applicable)
                # If this is a MonitorsChannel of multiple Monitors to plot
                if isinstance(channel, MonitorsChannel):
                    for monitor in channel.monitors:
                        if monitor.train_flag:
                            name = COLLAPSE_SEPARATOR.join([channel.name, monitor.name, TRAIN_MARKER])
                            self._create_line(fig, name)

                        if monitor.valid_flag:
                            name = COLLAPSE_SEPARATOR.join([channel.name, monitor.name, VALID_MARKER])
                            self._create_line(fig, name)

                        if monitor.test_flag:
                            name = COLLAPSE_SEPARATOR.join([channel.name, monitor.name, TEST_MARKER])
                            self._create_line(fig, name)

                # otherwise it is a single Monitor
                else:
                    if channel.train_flag:
                        name = COLLAPSE_SEPARATOR.join([channel.name, TRAIN_MARKER])
                        self._create_line(fig, name)

                    if channel.valid_flag:
                        name = COLLAPSE_SEPARATOR.join([channel.name, VALID_MARKER])
                        self._create_line(fig, name)

                    if channel.test_flag:
                        name = COLLAPSE_SEPARATOR.join([channel.name, TEST_MARKER])
                        self._create_line(fig, name)

            if open_browser:
                session.show()

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
                if key in self.data_sources:
                    self.data_sources[key].stream({'x':[epoch], 'y':[value]})
                else:
                    log.warning("Monitor named %s not found in the plot!" % key)

    def _create_line(self, fig, name):
        # create a new line
        name_without_fig = name.split(COLLAPSE_SEPARATOR, 1)[1]
        line = fig.line([], [], legend=name_without_fig, name=name_without_fig,
                        line_color=self.colors[self.line_color_idx % len(self.colors)])
        self.line_color_idx += 1
        self.data_sources[name] = line.data_source
