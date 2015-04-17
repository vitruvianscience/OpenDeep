__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

import theano
import theano.tensor as T
import collections
import time
from collections import OrderedDict
from opendeep.utils.plot import Plot
from opendeep.utils.noise import add_uniform
from opendeep.utils.statistics import get_stats
from opendeep.utils.misc import make_time_units_string

def main():
    var = theano.shared(T.zeros(shape=(88,100), dtype=theano.config.floatX).eval(), name='W')
    updates = [(var, add_uniform(input=var, interval=.2))]

    out = T.sum(var**2)

    stats = get_stats(var)
    stats.pop('l1')
    stats.pop('l2')
    # stats.pop('min')
    # stats.pop('max')
    stats.pop('var')
    stats.pop('std')
    stats.pop('mean')
    monitors = OrderedDict({var.name:stats})
    # monitors.update(get_stats(var))

    # monitors = OrderedDict(get_stats(var))
    # monitors = {'mean': T.mean(var)}

    monitors_collapsed = OrderedDict()
    for key, val in monitors.items():
        if isinstance(val, collections.Mapping):
            for key2, val2 in val.items():
                monitors_collapsed['_'.join([key, key2])] = val2
        else:
            monitors_collapsed[key] = val



    plot = Plot(bokeh_doc_name='test_plots', monitors=monitors, start_server=False, open_browser=True)

    print 'compiling...'
    f = theano.function(inputs=[], outputs=monitors_collapsed.values(), updates=updates)
    print 'done'

    t1=time.time()
    for epoch in range(500):
        t=time.time()
        print epoch
        vals = f()
        m = OrderedDict(zip(monitors_collapsed.keys(), vals))
        plot.update_plots(epoch, m)
        print '-----', make_time_units_string(time.time()-t)

    print
    print
    print "TOTAL TIME ", make_time_units_string(time.time()-t1)


if __name__ == '__main__':
    main()