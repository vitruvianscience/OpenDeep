#!/usr/bin/python
# standard imports
import time
# third-party imports
from theano import function, config, shared
import theano.tensor as T
import numpy


def main():
    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
    iters = 1000

    rng = numpy.random.RandomState(22)
    x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
    f = function([], T.exp(x))
    print f.maker.fgraph.toposort()
    t0 = time.time()
    for i in xrange(iters):
        r = f()
    t1 = time.time()
    print 'Looping %d times took' % iters, t1 - t0, 'seconds'
    print 'Result is', r
    if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
        print 'Used the cpu'
    else:
        print 'Used the gpu'

if __name__ == '__main__':
    main()
