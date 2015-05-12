import numpy
from opendeep import as_floatX
from opendeep.log.logger import config_root_logger
from opendeep.data.dataset import TRAIN, VALID, TEST
from opendeep.data.standard_datasets.image.mnist import MNIST
from opendeep.optimization.log_likelihood.conservative_sampling_ll import _compile_csl_fn, _compile_csl_fn_v2, \
    compute_CSL_with_minibatches, compute_CSL_with_minibatches_one_chain


def bernoulli_csl(switch=0):

    mnist = MNIST()
    train_x, _ = mnist.getSubset(TRAIN)
    valid_x, _ = mnist.getSubset(VALID)
    test_x, _  = mnist.getSubset(TEST)

    mnist_b = MNIST(binary=True)
    train_x_b, _ = mnist_b.getSubset(TRAIN)
    valid_x_b, _ = mnist_b.getSubset(VALID)
    test_x_b, _  = mnist_b.getSubset(TEST)

    means = as_floatX(test_x).eval()
    means = numpy.clip(a=means, a_min=1e-10, a_max=(1 - (1e-5)))
    #means = as_floatX(numpy.random.uniform(size=(10000,784))) * 0 + 0.5

    minibatches = as_floatX(test_x_b.reshape((1000, 10, 784))).eval()

    if switch:
        # when means is a matrix of (N,D), representing only 1 chain
        csl_fn = _compile_csl_fn_v2(means)
        compute_CSL_with_minibatches_one_chain(csl_fn, minibatches)
    else:
        # when means is a 3D tensor (N, K, D)
        # When there are N chains, each chain having K samples of dimension D
        chains = means.reshape(10, 100, 10, 784)
        csl_fn = _compile_csl_fn()
        compute_CSL_with_minibatches(csl_fn, minibatches, chains)

    del mnist
    del mnist_b

if __name__ == '__main__':
    config_root_logger()
    bernoulli_csl(switch=0)