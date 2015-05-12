import time
import logging
from theano.compat.python2x import OrderedDict

import theano
import theano.tensor as T
from opendeep.monitor.monitor import Monitor, MonitorsChannel, collapse_channels
from opendeep.monitor.out_service import FileService
from opendeep.data.dataset import TRAIN, VALID, TEST
from opendeep.utils.noise import add_uniform
from opendeep.utils.statistics import get_stats
from opendeep.utils.misc import make_time_units_string
from opendeep.log.logger import config_root_logger

log = logging.getLogger(__name__)


def main():
    var = theano.shared(T.zeros(shape=(88, 100), dtype=theano.config.floatX).eval(), name='W')
    updates = [(var, add_uniform(input=var, noise_level=.02))]

    stats = get_stats(var)
    l1 = stats.pop('l1')
    l2 = stats.pop('l2')
    min = stats.pop('min')
    max = stats.pop('max')
    var = stats.pop('var')
    std = stats.pop('std')
    mean = stats.pop('mean')

    mean_monitor = Monitor('mean', mean, train=True, valid=True, out_service=FileService('outs/mean.txt'))
    var_monitor = Monitor('var', var, out_service=FileService('outs/var.txt'))

    w_channel = MonitorsChannel('W', monitors=mean_monitor)

    stat_channel = MonitorsChannel('stats', monitors=[var_monitor])

    monitors = [w_channel, stat_channel]

    train_collapsed_raw = collapse_channels(monitors, train=True)
    train_collapsed = OrderedDict([(item[0], item[1]) for item in train_collapsed_raw])
    train_services = OrderedDict([(item[0], item[2]) for item in train_collapsed_raw])
    valid_collapsed_raw = collapse_channels(monitors, valid=True)
    valid_collapsed = OrderedDict([(item[0], item[1]) for item in valid_collapsed_raw])
    valid_services = OrderedDict([(item[0], item[2]) for item in valid_collapsed_raw])

    log.debug('compiling...')
    f = theano.function(inputs=[], outputs=train_collapsed.values(), updates=updates)
    f2 = theano.function(inputs=[], outputs=valid_collapsed.values(), updates=updates)
    log.debug('done')

    t1=time.time()

    for epoch in range(10):
        t=time.time()
        log.debug(epoch)
        vals = f()
        m = OrderedDict(zip(train_collapsed.keys(), vals))
        for name, service in train_services.items():
            if name in m:
                service.write(m[name], TRAIN)
        log.debug('----- '+make_time_units_string(time.time()-t))

    for epoch in range(10):
        t = time.time()
        log.debug(epoch)
        vals = f2()
        m = OrderedDict(zip(valid_collapsed.keys(), vals))
        for name, service in valid_services.items():
            if name in m:
                service.write(m[name], VALID)
        log.debug('----- ' + make_time_units_string(time.time() - t))

    log.debug("TOTAL TIME "+make_time_units_string(time.time()-t1))


if __name__ == '__main__':
    config_root_logger()
    main()