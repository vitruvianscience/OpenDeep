# import data_provider
import numpy

import theano
import theano.tensor as T

from opendeep.old import data_tools


'''
Code from Li Yao (University of Montreal)
'''

def log_sum_exp_theano(x, axis):
    max_x = T.max(x, axis)
    return max_x + T.log(T.sum(T.exp(x - T.shape_padright(max_x, 1)), axis))

class CSL(object):
    def get_CSL_fn_independent_Bernoulli_v2(self, mu):
        '''
         mu is (N,D) numpy array
        p(x) = sum_h p(x|h)p(h) where p(x|h) is independent Bernoulli with
        a vector mu, mu_i for dim_i
        '''
        #
        print 'building theano fn for Bernoulli CSL (v2)'
        x = T.fmatrix('inputs')
        x.tag.test_value = numpy.random.uniform(size=(10,784)).astype('float32')
        mu = numpy.clip(mu,1e-10,(1-(1e-5)))
        mu = mu[None,:,:]
        inner_1 = numpy.log(mu)
        inner_2 = numpy.log(numpy.float32(1)-mu)

        k = mu.shape[1]
        D = mu.shape[2]

        # there are two terms in the log(p(x|mu))

        term_1 = -T.log(k)
        c = T.sum(x.dimshuffle(0,'x',1) * inner_1 +
                  (numpy.float32(1)-x.dimshuffle(0,'x',1))*inner_2,
                  axis=2)
        debug = c.sum(axis=1)
        term_2 = log_sum_exp_theano(c, axis=1)

        log_likelihood = term_1 + term_2
        f = theano.function([x], log_likelihood)
        return f
        
        
    def get_CSL_fn_independent_Bernoulli(self):
        '''
        BUG HERE, not doing properly by chains (still has the bug, I don't see it)
        This is taking too much GPU mem

        mean: N(# of chains)*K(samples per chain)*D(data dim)
        minibatch: M(# of examples)*D (data dim)

        return: M * N matrix where each element is LL of one example against
        one chain.
        
        '''
        print 'building theano fn for Bernoulli CSL (original)'
        means = T.ftensor3('chains')
        minibatch = T.fmatrix('inputs')

        # how many chains CSL average over
        N = 5
        # minibatch size
        M = 10
        # data dim
        D = 784
        minibatch.tag.test_value = numpy.random.binomial(1,0.5,size=(M,D)).astype('float32')
        # chain length
        K = 100
        means.tag.test_value = numpy.random.uniform(size=(N,K,D)).astype('float32')
        
        # computing LL
        
        # the length of each chain
        sample_size = means.shape[1]
        
        _minibatch = minibatch.dimshuffle(0,'x', 'x', 1)
        _means = means.dimshuffle('x',0,1,2)
        
        A = T.log(sample_size)
        B = _minibatch * T.log(_means) + (numpy.float32(1)-_minibatch) \
            * T.log(numpy.float32(1)-_means)
        C = B.sum(axis=3)
        D = log_sum_exp_theano(C, axis=2)
        E = D - A
        #G = E.mean(axis=1)
        f = theano.function(
            inputs=[minibatch, means],
            outputs=E,
            name='CSL_independent_bernoulli_fn'
        )

        return f

    def compute_CSL_with_minibatches_one_chain(self, fn, minibatches):
        LLs = []
        for i, minibatch in enumerate(minibatches):
            # loop through one minibatch
            LL = fn(minibatch)
            LLs.append(LL)
            t = numpy.mean(LLs)
            print '%d  /  %d batches, LL mean so far %.4f'%(i+1, minibatches.shape[0], t)
        print 'mean LL', numpy.mean(LLs)
        
    def compute_CSL_with_minibatches(self, fn, minibatches, chains):
        # fn is the compiled theano fn
        LLs = []
        for i, minibatch in enumerate(minibatches):
            # loop through one minibatch
            LL_minibatch_all_chains = []
            for chain_minibatch in chains:
                # loop through a minibatch of chains
                LL = fn(minibatch, chain_minibatch)
                LL_minibatch_all_chains.append(LL)
            LL_minibatch_all_chains = numpy.concatenate(LL_minibatch_all_chains,axis=1)
            #import ipdb; ipdb.set_trace()
            LLs.append(LL_minibatch_all_chains)
            t = numpy.mean(LLs)
            print '%d  /  %d batches, LL mean so far %.4f'%(i+1, minibatches.shape[0], t)
        LLs = numpy.concatenate(LLs,axis=0)
        print 'mean LL ', LLs.mean()
        return 

def test_bernoulli_csl():
    print 'loading MNIST test set'
#     (train_x, _,
#      valid_x, _,
#      test_x, _) = data_provider.load_mnist(binary=False, standard_split=False)
#     (train_x_b, _,
#      valid_x_b, _,
#      test_x_b, _) = data_provider.load_mnist(binary=True, standard_split=False)
    (_,_), (_,_), (test_x,_) = data_tools.load_mnist('../../data')
    (_,_), (_,_), (test_x_b,_) = data_tools.load_mnist_binary('../../data')
    
    means = test_x.astype('float32')
    means = numpy.clip(means,1e-10,(1-(1e-5)))
    #means = numpy.random.uniform(size=(10000,784)).astype('float32') * 0 + 0.5
    
    csl = CSL()
    minibatches = test_x_b.reshape((1000,10,784)).astype('float32')
    
    if 0:
        # when means is a matrix of (N,D), representing only 1 chain
        csl_fn = csl.get_CSL_fn_independent_Bernoulli_v2(means)
        csl.compute_CSL_with_minibatches_one_chain(csl_fn, minibatches)
    else:
        # when means is a 3D tensor (N, K, D)
        # When there are N chains, each chain having K samples of dimension D
        chains = means.reshape(10,100,10,784)
        csl_fn = csl.get_CSL_fn_independent_Bernoulli()
        csl.compute_CSL_with_minibatches(csl_fn, minibatches, chains)
    

if __name__ == '__main__':
    test_bernoulli_csl()
    
