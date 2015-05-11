"""
Implements the Conservative Sampling-based Log-likelihood estimator. This is useful for generative model comparison.

From the paper:
"Bounding the Test Log-Likelihood of Generative Models"
Yoshua Bengio, Li Yao, Kyunghyun Cho
http://arxiv.org/pdf/1311.6184v4.pdf

From Li Yao:
The idea of CSL is simple:
log p(x) = log Eh p(x|h) >= Eh log p(x|h) where h are sampled from GSN you have trained.

So when you have a trained GSN, p_theta (x|h) is parameterized by theta. So given one particular test example x0,
and sampled h, you can easily compute p_theta (x0|h).

In order for CSL to work well, samples of h must come from the true p(h).
However, this is never the case with limited number of h taken from a Markov chain.
What I did in the paper is to take one h every K steps. The assumption being made here is that samples
of h from higher layers mix much faster then x, which is what we usually observed in practice.
"""
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Li Yao, Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
import time
# third party
import numpy
import theano.tensor as T
# internal
from opendeep import function, as_floatX
from opendeep.utils.misc import make_time_units_string

log = logging.getLogger(__name__)


def log_sum_exp_theano(x, axis):
    """
    Calculate T.max(x, axis) + T.log(T.sum(T.exp(x - T.shape_padright(max_x, 1)), axis))
    """
    max_x = T.max(x, axis)
    return max_x + T.log(T.sum(T.exp(x - T.shape_padright(max_x, 1)), axis))


def _compile_csl_fn():
    """
    BUG HERE, not doing properly by chains (still has the bug, I don't see it)
    This is taking too much GPU mem

    mean: N(# of chains)*K(samples per chain)*D(data dim)
    minibatch: M(# of examples)*D (data dim)
    M * N matrix where each element is LL of one example against one chain.

    This function is for computing CSL over parallel chains of minibatches.

    Returns
    -------
    theano function
        Function computing M * N matrix where each element is LL of one example against one chain.
    """
    # when means is a 3D tensor (N, K, D)
    # When there are N chains, each chain having K samples of dimension D
    log.debug('building theano fn for Bernoulli CSL')
    means = T.tensor3('chains')
    minibatch = T.matrix('inputs')

    # how many chains CSL average over
    N = 5
    # minibatch size
    M = 10
    # data dim
    D = 784
    minibatch.tag.test_value = as_floatX(numpy.random.binomial(1, 0.5, size=(M, D)))
    # chain length
    K = 100
    means.tag.test_value = as_floatX(numpy.random.uniform(size=(N, K, D)))

    # computing LL

    # the length of each chain
    sample_size = means.shape[1]

    _minibatch = minibatch.dimshuffle(0, 'x', 'x', 1)
    _means = means.dimshuffle('x', 0, 1, 2)

    A = T.log(sample_size)
    B = _minibatch * T.log(_means) + (1. - _minibatch) * T.log(1. - _means)
    C = B.sum(axis=3)
    D = log_sum_exp_theano(C, axis=2)
    E = D - A
    # G = E.mean(axis=1)
    f = function(
        inputs=[minibatch, means],
        outputs=E,
        name='CSL_independent_bernoulli_fn'
    )
    return f


def _compile_csl_fn_v2(mu):
    """
    p(x) = sum_h p(x|h)p(h) where p(x|h) is independent Bernoulli with
    a vector mu, mu_i for dim_i

    This function is for computing CSL over minibatches (in a single chain).

    Parameters
    ----------
    mu : array_like
        mu is (N,D) numpy array

    Returns
    -------
    theano function
        Function computing the Bernoulli CSL log likelihood.
    """
    #
    log.debug('building theano fn for Bernoulli CSL')
    x = T.fmatrix('inputs')
    x.tag.test_value = as_floatX(numpy.random.uniform(size=(10, 784)))
    mu = numpy.clip(mu, 1e-10, (1 - (1e-5)))
    mu = mu[None, :, :]
    inner_1 = numpy.log(mu)
    inner_2 = numpy.log(1. - mu)

    k = mu.shape[1]
    D = mu.shape[2]

    # there are two terms in the log(p(x|mu))

    term_1 = -T.log(k)
    c = T.sum(x.dimshuffle(0, 'x', 1) * inner_1 +
              (1. - x.dimshuffle(0, 'x', 1)) * inner_2,
              axis=2)
    debug = c.sum(axis=1)
    term_2 = log_sum_exp_theano(c, axis=1)

    log_likelihood = term_1 + term_2
    f = function([x], log_likelihood, name='CSL_independent_bernoulli_fn')
    return f


def compute_CSL_with_minibatches_one_chain(fn, minibatches):
    """
    Computes the CSL over minibatches with a single chain (no parallel chains to average computation over).

    Parameters
    ----------
    fn : theano function
        The CSL function to use.
    minibatches : tensor
        The minibatches of data as a 3D tensor with shape (num_minibatches, batch_size, input_dimensionality).

    Returns
    -------
    float
        The mean LL value over minibatches.
    """
    LLs = []
    t = time.time()
    mean = None
    for i, minibatch in enumerate(minibatches):
        # loop through one minibatch
        LL = fn(minibatch)
        LLs.append(LL)
        mean = numpy.mean(LLs)
        log.info('%d  /  %d batches, LL mean so far %.4f' % (i + 1, minibatches.shape[0], mean))
    log.info('mean LL %s' % mean)
    log.info('--- took %s ---' % make_time_units_string(time.time() - t))
    return mean


def compute_CSL_with_minibatches(fn, minibatches, chains):
    """
    Computes CSL over parallel chains of minibatches (means the input chains is a 4D tensor consisting of minibatches
    of shape (N, K, D)). When there are N chains, each chain having K samples of dimension D

    Parameters
    ----------
    fn : theano function
        The CSL function to use.
    minibatches : tensor
        The minibatches of data as a 3D tensor with shape (num_minibatches, batch_size, input_dimensionality).
    chains : tensor
        The chains of data as a 4D tensor with shape (n_minibatches, n_chains, batch_size, input_dimensionality).
    Returns
    -------
    float
        The mean LL value over minibatches.
    """
    # fn is the compiled theano fn
    LLs = []
    t = time.time()
    for i, minibatch in enumerate(minibatches):
        # loop through one minibatch
        LL_minibatch_all_chains = []
        for chain_minibatch in chains:
            # loop through a minibatch of chains
            LL = fn(minibatch, chain_minibatch)
            LL_minibatch_all_chains.append(LL)
        LL_minibatch_all_chains = numpy.concatenate(LL_minibatch_all_chains, axis=1)
        # import ipdb; ipdb.set_trace()
        LLs.append(LL_minibatch_all_chains)
        mean = numpy.mean(LLs)
        log.info('%d  /  %d batches, LL mean so far %.4f' % (i + 1, minibatches.shape[0], mean))
    LLs = numpy.concatenate(LLs, axis=0)
    mean_LLs = LLs.mean()
    log.info('mean LL %s' % str(mean_LLs))
    log.info('--- took %s ---' % make_time_units_string(time.time() - t))
    return mean_LLs