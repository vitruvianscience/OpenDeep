__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

import time
import logging
from theano.compat.python2x import OrderedDict

import theano
import theano.tensor as T

from opendeep.monitor.monitor import Monitor, MonitorsChannel, collapse_channels, values_dict_from_collapsed
from opendeep.monitor.plot import Plot
from opendeep.utils.noise import add_uniform
from opendeep.utils.statistics import get_stats
from opendeep.utils.misc import make_time_units_string
from opendeep.log.logger import config_root_logger

log = logging.getLogger(__name__)


def main():
    var = theano.shared(T.zeros(shape=(88, 100), dtype=theano.config.floatX).eval(), name='W')
    updates = [(var, add_uniform(input=var, interval=.02))]

    stats = get_stats(var)
    l1 = stats.pop('l1')
    l2 = stats.pop('l2')
    min = stats.pop('min')
    max = stats.pop('max')
    var = stats.pop('var')
    std = stats.pop('std')
    mean = stats.pop('mean')

    mean_monitor = Monitor('mean', mean)
    var_monitor = Monitor('var', var)

    w_channel = MonitorsChannel('W', monitors=mean_monitor)

    stat_channel = MonitorsChannel('stats', monitors=[var_monitor])

    monitors = [w_channel, stat_channel]

    all_collapsed = OrderedDict(collapse_channels(monitors))

    plot = Plot(bokeh_doc_name='test_plots', channels=monitors, start_server=False, open_browser=True)

    log.debug('compiling...')
    f = theano.function(inputs=[], outputs=all_collapsed.values(), updates=updates)
    log.debug('done')

    t1=time.time()
    for epoch in range(500):
        t=time.time()
        log.debug(epoch)
        vals = f()
        m = values_dict_from_collapsed(all_collapsed.keys(), vals)
        plot.update_plots(epoch, m)
        log.debug('----- '+make_time_units_string(time.time()-t))

    log.debug("TOTAL TIME "+make_time_units_string(time.time()-t1))


if __name__ == '__main__':
    config_root_logger()
    main()