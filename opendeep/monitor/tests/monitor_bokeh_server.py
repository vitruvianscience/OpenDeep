import time
import logging
from theano.compat.python2x import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from bokeh.client import push_session
from bokeh.driving import cosine
from bokeh.plotting import (curdoc, output_server, figure)
from bokeh.models.renderers import GlyphRenderer

from opendeep.monitor.monitor import Monitor, MonitorsChannel, collapse_channels
from opendeep.monitor.plot import Plot
from opendeep.utils.noise import add_uniform
from opendeep.utils.statistics import get_stats
from opendeep.utils.misc import make_time_units_string, raise_to_list
from opendeep.log.logger import config_root_logger

log = logging.getLogger(__name__)


def main():
    w = theano.shared(T.zeros(shape=(88, 100), dtype=theano.config.floatX).eval(), name='W')
    updates = [(w, add_uniform(input=w, noise_level=.02))]

    stats = get_stats(w)
    l1 = stats.pop('l1')
    l2 = stats.pop('l2')
    min = stats.pop('min')
    max = stats.pop('max')
    var = stats.pop('var')
    std = stats.pop('std')
    mean = stats.pop('mean')

    mean_monitor = Monitor('mean', mean, train=True, valid=True)
    stat_monitor = Monitor('max', max)

    w_channel = MonitorsChannel('W', monitors=mean_monitor)

    stat_channel = MonitorsChannel('stats', monitors=[stat_monitor])

    monitors = [w_channel, stat_channel]

    train_collapsed = collapse_channels(monitors, train=True)
    train_collapsed = OrderedDict([(name, expression) for name, expression, _ in train_collapsed])
    valid_collapsed = collapse_channels(monitors, valid=True)
    valid_collapsed = OrderedDict([(name, expression) for name, expression, _ in valid_collapsed])

    plot = Plot(bokeh_doc_name='test_plots', monitor_channels=monitors, open_browser=True)

    log.debug('compiling...')
    f = theano.function(inputs=[], outputs=list(train_collapsed.values()), updates=updates)
    f2 = theano.function(inputs=[], outputs=list(valid_collapsed.values()), updates=updates)
    log.debug('done')

    t1=time.time()

    for epoch in range(100):
        t=time.time()
        log.debug(epoch)
        vals = f()
        m = OrderedDict(zip(train_collapsed.keys(), vals))
        plot.update_plots(epoch, m)
        time.sleep(0.02)
        log.debug('----- '+make_time_units_string(time.time()-t))

    for epoch in range(100):
        t = time.time()
        log.debug(epoch)
        vals = f2()
        m = OrderedDict(zip(valid_collapsed.keys(), vals))
        plot.update_plots(epoch, m)
        time.sleep(0.02)
        log.debug('----- ' + make_time_units_string(time.time() - t))

    log.debug("TOTAL TIME "+make_time_units_string(time.time()-t1))

def test_server():
    session = push_session(curdoc())
    session.show()
    # create the figure
    fig = figure(title='testing',
                 x_axis_label='iterations',
                 y_axis_label='value',
                 logo=None,
                 toolbar_location='right')
    # create a new line
    l = fig.line([], [], legend='test_line', name='test_line',
                line_color='#1f77b4')

    for i in range(100):
        if i==0:
            l.data_source.stream({'x':[i],'y':[10]})
        else:
            l.data_source.stream({'x':[i],'y':[i]})
        time.sleep(0.05)
    print 'done!'




if __name__ == '__main__':
    config_root_logger()
    main()
    # test_server()
