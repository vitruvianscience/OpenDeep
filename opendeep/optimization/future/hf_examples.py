# Author: Nicolas Boulanger-Lewandowski
# University of Montreal, 2012-2013


import numpy
import sys

import theano
import theano.tensor as T

from opendeep.optimization.future.hessian_free import hf_optimizer, SequenceDataset


def test_cg(n=500):
  '''Attempt to solve a linear system using the CG function in hf_optimizer.'''

  A = numpy.random.uniform(-1, 1, (n, n))
  A = numpy.dot(A.T, A)
  val, vec = numpy.linalg.eig(A)
  val = numpy.random.uniform(1, 5000, (n, 1))
  A = numpy.dot(vec.T, val*vec)

  # hack into a fake hf_optimizer object
  x = theano.shared(0.0)
  s = 2.0*x
  hf = hf_optimizer([x], [], s, [s**2])
  hf.quick_cost = lambda *args, **kwargs: 0.0
  hf.global_backtracking = False
  hf.preconditioner = False
  hf.max_cg_iterations = 300
  hf.batch_Gv = lambda v: numpy.dot(A, v)
  b = numpy.random.random(n)
  c, x, j, i = hf.cg(b)
  print

  print 'error on b =', abs(numpy.dot(A, x) - b).mean()
  print 'error on x =', abs(numpy.linalg.solve(A, b) - x).mean()


def sgd_optimizer(p, inputs, costs, train_set, lr=1e-4):
  '''SGD optimizer with a similar interface to hf_optimizer.'''

  g = [T.grad(costs[0], i) for i in p]
  updates = dict((i, i - lr*j) for i, j in zip(p, g))
  f = theano.function(inputs, costs, updates=updates)
  
  try:
    for u in xrange(1000):
      cost = []
      for i in train_set.iterate(True):
        cost.append(f(*i))
      print 'update %i, cost=' %u, numpy.mean(cost, axis=0)
      sys.stdout.flush()

  except KeyboardInterrupt: 
    print 'Training interrupted.'


# feed-forward neural network with sigmoidal output
def simple_NN(sizes=(784, 100, 10)):
  x = T.matrix()
  t = T.matrix()

  p = []
  y = x

  for i in xrange(len(sizes)-1):
    a, b = sizes[i:i+2]
    Wi = theano.shared((10./numpy.sqrt(a+b) * numpy.random.uniform(-1, 1, size=(a, b))).astype(theano.config.floatX))
    bi = theano.shared(numpy.zeros(b, dtype=theano.config.floatX))
    p += [Wi, bi]

    s = T.dot(y,Wi) + bi
    y = T.nnet.sigmoid(s)

  c = (-t* T.log(y) - (1-t)* T.log(1-y)).mean()
  acc = T.neq(T.round(y), t).mean()

  return p, [x, t], s, [c, acc]


def example_NN(hf=True):
  p, inputs, s, costs = simple_NN((2, 50, 40, 30, 1))

  xor_dataset = [[], []]
  for i in xrange(50000):
    x = numpy.random.randint(0, 2, (50, 2))
    t = (x[:, 0:1] ^ x[:, 1:2]).astype(theano.config.floatX)
    x = x.astype(theano.config.floatX)
    xor_dataset[0].append(x)
    xor_dataset[1].append(t)

  training_examples = len(xor_dataset[0]) * 3/4
  train = [xor_dataset[0][:training_examples], xor_dataset[1][:training_examples]]
  valid = [xor_dataset[0][training_examples:], xor_dataset[1][training_examples:]]

  gradient_dataset = SequenceDataset(train, batch_size=None, number_batches=10000)
  cg_dataset = SequenceDataset(train, batch_size=None, number_batches=5000)
  valid_dataset = SequenceDataset(valid, batch_size=None, number_batches=5000)
  
  if hf:
    hf_optimizer(p, inputs, s, costs).train(gradient_dataset, cg_dataset, initial_lambda=1.0, preconditioner=True, validation=valid_dataset)
  else:
    sgd_optimizer(p, inputs, costs, gradient_dataset, lr=1e-3)
    

# single-layer recurrent neural network with sigmoid output, only last time-step output is significant
def simple_RNN(nh):
  Wx = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (1, nh)).astype(theano.config.floatX))
  Wh = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh)).astype(theano.config.floatX))
  Wy = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (nh, 1)).astype(theano.config.floatX))
  bh = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
  by = theano.shared(numpy.zeros(1, dtype=theano.config.floatX))
  h0 = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
  p = [Wx, Wh, Wy, bh, by, h0]

  x = T.matrix()

  def recurrence(x_t, h_tm1):
    ha_t = T.dot(x_t, Wx) + T.dot(h_tm1, Wh) + bh
    h_t = T.tanh(ha_t)
    s_t = T.dot(h_t, Wy) + by
    return [ha_t, h_t, s_t]

  ([ha, h, activations], updates) = theano.scan(fn=recurrence, sequences=x, outputs_info=[dict(), h0, dict()])

  h = T.tanh(ha)  # so it is differentiable with respect to ha
  t = x[0, 0]
  s = activations[-1, 0]
  y = T.nnet.sigmoid(s)
  loss = -t*T.log(y + 1e-14) - (1-t)*T.log((1-y) + 1e-14)
  acc = T.neq(T.round(y), t)
  
  return p, [x], s, [loss, acc], h, ha


def example_RNN(hf=True):
  p, inputs, s, costs, h, ha = simple_RNN(100)

  memorization_dataset = [[]]  # memorize the first unit for 100 time-steps with binary noise
  for i in xrange(100000):
    memorization_dataset[0].append(numpy.random.randint(2, size=(100, 1)).astype(theano.config.floatX))

  train = [memorization_dataset[0][:-1000]]
  valid = [memorization_dataset[0][-1000:]]
  
  gradient_dataset = SequenceDataset(train, batch_size=None, number_batches=5000)
  cg_dataset = SequenceDataset(train, batch_size=None, number_batches=1000)
  valid_dataset = SequenceDataset(valid, batch_size=None, number_batches=1000)

  if hf:
    hf_optimizer(p, inputs, s, costs, 0.5*(h + 1), ha).train(gradient_dataset, cg_dataset, initial_lambda=0.5, mu=1.0, preconditioner=False, validation=valid_dataset)
  else:
    sgd_optimizer(p, inputs, costs, gradient_dataset, lr=5e-5)    


