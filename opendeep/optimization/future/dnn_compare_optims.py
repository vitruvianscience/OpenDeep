"""
A deep neural network with or w/o dropout in one file.
"""

import numpy
import theano
import sklearn
import sys
import math
from theano import tensor as T
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict


BATCH_SIZE = 1


def relu_f(vec):
    """ Wrapper to quickly change the rectified linear unit function """
    return (vec + abs(vec)) / 2.


def dropout(rng, x, p=0.5):
    """ Zero-out random values in x with probability p using rng """
    if p > 0. and p < 1.:
        seed = rng.randint(2 ** 30)
        srng = theano.tensor.shared_randomstreams.RandomStreams(seed)
        mask = srng.binomial(n=1, p=1.-p, size=x.shape,
                dtype=theano.config.floatX)
        return x * mask
    return x


def build_shared_zeros(shape, name):
    """ Builds a theano shared variable filled with a zeros numpy array """
    return shared(value=numpy.zeros(shape, dtype=theano.config.floatX),
            name=name, borrow=True)


class Linear(object):
    """ Basic linear transformation layer (W.X + b) """
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            W_values *= 4  # This works for sigmoid activated networks!
            W = theano.shared(value=W_values, name='W', borrow=True)
        if b is None:
            b = build_shared_zeros((n_out,), 'b')
        self.input = input
        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        self.output = T.dot(self.input, self.W) + self.b

    def __repr__(self):
        return "Linear"


class SigmoidLayer(Linear):
    """ Sigmoid activation layer (sigmoid(W.X + b)) """
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        super(SigmoidLayer, self).__init__(rng, input, n_in, n_out, W, b)
        self.pre_activation = self.output
        self.output = T.nnet.sigmoid(self.pre_activation)


class ReLU(Linear):
    """ Rectified Linear Unit activation layer (max(0, W.X + b)) """
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        if b is None:
            b = build_shared_zeros((n_out,), 'b')
        super(ReLU, self).__init__(rng, input, n_in, n_out, W, b)
        self.pre_activation = self.output
        self.output = relu_f(self.pre_activation)


class DatasetMiniBatchIterator(object):
    """ Basic mini-batch iterator """
    def __init__(self, x, y, batch_size=BATCH_SIZE, randomize=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.randomize = randomize
        from sklearn.utils import check_random_state
        self.rng = check_random_state(42)

    def __iter__(self):
        n_samples = self.x.shape[0]
        if self.randomize:
            for _ in xrange(n_samples / BATCH_SIZE):
                if BATCH_SIZE > 1:
                    i = int(math.floor(self.rng.rand(1) * (n_samples / BATCH_SIZE + 1)))
                else:
                    i = int(math.floor(self.rng.rand(1) * n_samples))
                yield (i, self.x[i*self.batch_size:(i+1)*self.batch_size],
                       self.y[i*self.batch_size:(i+1)*self.batch_size])
        else:
            for i in xrange((n_samples + self.batch_size - 1)
                            / self.batch_size):
                yield (self.x[i*self.batch_size:(i+1)*self.batch_size],
                       self.y[i*self.batch_size:(i+1)*self.batch_size])


class LogisticRegression:
    """Multi-class Logistic Regression
    """
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        if W != None:
            self.W = W
        else:
            self.W = build_shared_zeros((n_in, n_out), 'W')
        if b != None:
            self.b = b
        else:
            self.b = build_shared_zeros((n_out,), 'b')

        # P(Y|X) = softmax(W.X + b)
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.output = self.y_pred
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def negative_log_likelihood_sum(self, y):
        return -T.sum(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def training_cost(self, y):
        """ Wrapper for standard name """
        return self.negative_log_likelihood_sum(y)

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError("y should have the same shape as self.y_pred",
                ("y", y.type, "y_pred", self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            print("!!! y should be of int type")
            return T.mean(T.neq(self.y_pred, numpy.asarray(y, dtype='int')))


class NeuralNet(object):
    """ Neural network (not regularized, without dropout) """
    def __init__(self, numpy_rng, theano_rng=None, 
                 n_ins=40*3,
                 layers_types=[Linear, ReLU, ReLU, ReLU, LogisticRegression],
                 layers_sizes=[1024, 1024, 1024, 1024],
                 n_outs=62 * 3,
                 rho=0.9, eps=1.E-6,
                 debugprint=False):
        """
        TODO
        """
        self.layers = []
        self.params = []
        self.pre_activations = [] # SAG specific
        self.n_layers = len(layers_types)
        self.layers_types = layers_types
        assert self.n_layers > 0
        self._rho = rho  # ``momentum'' for adadelta
        self._eps = eps  # epsilon for adadelta
        self._accugrads = []  # for adadelta
        self._accudeltas = []  # for adadelta
        self._sag_gradient_memory = []  # for SAG

        if theano_rng == None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.x = T.fmatrix('x')
        self.y = T.ivector('y')
        
        self.layers_ins = [n_ins] + layers_sizes
        self.layers_outs = layers_sizes + [n_outs]
        
        layer_input = self.x
        
        for layer_type, n_in, n_out in zip(layers_types,
                self.layers_ins, self.layers_outs):
            this_layer = layer_type(rng=numpy_rng,
                    input=layer_input, n_in=n_in, n_out=n_out)
            assert hasattr(this_layer, 'output')
            self.params.extend(this_layer.params)
            #self.pre_activations.extend(this_layer.pre_activation)# SAG specific TODO 
            self._accugrads.extend([build_shared_zeros(t.shape.eval(),
                'accugrad') for t in this_layer.params])
            self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
                'accudelta') for t in this_layer.params])

            self._sag_gradient_memory.extend([build_shared_zeros(tuple([x_train.shape[0] / BATCH_SIZE + 1] + list(t.shape.eval())), 'sag_gradient_memory') for t in this_layer.params])
            #self._sag_gradient_memory.extend([[build_shared_zeros(t.shape.eval(), 'sag_gradient_memory') for _ in xrange(x_train.shape[0] / BATCH_SIZE + 1)] for t in this_layer.params])

            self.layers.append(this_layer)
            layer_input = this_layer.output

        assert hasattr(self.layers[-1], 'training_cost')
        assert hasattr(self.layers[-1], 'errors')
        # TODO standardize cost
        self.mean_cost = self.layers[-1].negative_log_likelihood(self.y)
        self.cost = self.layers[-1].training_cost(self.y)
        if debugprint:
            theano.printing.debugprint(self.cost)

        self.errors = self.layers[-1].errors(self.y)

    def __repr__(self):
        dimensions_layers_str = map(lambda x: "x".join(map(str, x)),
                                    zip(self.layers_ins, self.layers_outs))
        return "_".join(map(lambda x: "_".join((x[0].__name__, x[1])),
                            zip(self.layers_types, dimensions_layers_str)))


    def get_SGD_trainer(self):
        """ Returns a plain SGD minibatch trainer with learning rate as param.
        """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        learning_rate = T.fscalar('lr')  # learning rate to use
        # compute the gradients with respect to the model parameters
        # using mean_cost so that the learning rate is not too dependent
        # on the batch size
        gparams = T.grad(self.mean_cost, self.params)

        # compute list of weights updates
        updates = OrderedDict()
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - gparam * learning_rate

        train_fn = theano.function(inputs=[theano.Param(batch_x),
                                           theano.Param(batch_y),
                                           theano.Param(learning_rate)],
                                   outputs=self.mean_cost,
                                   updates=updates,
                                   givens={self.x: batch_x, self.y: batch_y})

        return train_fn

    def get_SAG_trainer(self, R=1., alpha=0., debug=False):  # alpha for reg.
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        ind_minibatch = T.iscalar('ind_minibatch')
        n_seen = T.fscalar('n_seen')
        # compute the gradients with respect to the model parameters
        cost = self.cost
        gparams = T.grad(cost, self.params)
        #sparams = T.grad(cost, self.pre_activations)  # SAG specific

        scaling = numpy.float32(1. / (R / 4. + alpha))

        updates = OrderedDict()
        for accugrad, gradient_memory, param, gparam in zip(
                self._accugrads, self._sag_gradient_memory,
                #self._accugrads, self._sag_gradient_memory[ind_minibatch.eval()],
                self.params, gparams):
            new = gparam + alpha * param
            agrad = accugrad + new - gradient_memory[ind_minibatch]
            # updates[gradient_memory[ind_minibatch]] = new
            updates[gradient_memory] = T.set_subtensor(gradient_memory[ind_minibatch], new)

            updates[param] = param - (scaling / n_seen) * agrad
            updates[accugrad] = agrad

        train_fn = theano.function(inputs=[theano.Param(batch_x), 
            theano.Param(batch_y), theano.Param(ind_minibatch),
            theano.Param(n_seen)],
            outputs=self.cost,
            updates=updates,
            givens={self.x: batch_x, self.y: batch_y})

        return train_fn

    def get_adagrad_trainer(self):
        """ Returns an Adagrad (Duchi et al. 2010) trainer using a learning rate.
        """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        learning_rate = T.fscalar('lr')  # learning rate to use
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.cost, self.params)

        # compute list of weights updates
        updates = OrderedDict()
        for accugrad, param, gparam in zip(self._accugrads, self.params, gparams):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = accugrad + gparam * gparam
            dx = - (learning_rate / T.sqrt(agrad + self._eps)) * gparam
            updates[param] = param + dx
            updates[accugrad] = agrad

        train_fn = theano.function(inputs=[theano.Param(batch_x), 
            theano.Param(batch_y),
            theano.Param(learning_rate)],
            outputs=self.cost,
            updates=updates,
            givens={self.x: batch_x, self.y: batch_y})

        return train_fn

    def get_adadelta_trainer(self):
        """ Returns an Adadelta (Zeiler 2012) trainer using self._rho and
        self._eps params.
        """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.cost, self.params)

        # compute list of weights updates
        updates = OrderedDict()
        for accugrad, accudelta, param, gparam in zip(self._accugrads,
                self._accudeltas, self.params, gparams):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = self._rho * accugrad + (1 - self._rho) * gparam * gparam
            dx = - T.sqrt((accudelta + self._eps)
                          / (agrad + self._eps)) * gparam
            updates[accudelta] = (self._rho * accudelta
                                  + (1 - self._rho) * dx * dx)
            updates[param] = param + dx
            updates[accugrad] = agrad

        train_fn = theano.function(inputs=[theano.Param(batch_x),
                                           theano.Param(batch_y)],
                                   outputs=self.cost,
                                   updates=updates,
                                   givens={self.x: batch_x, self.y: batch_y})

        return train_fn

    def score_classif(self, given_set):
        """ Returns functions to get current classification errors. """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        score = theano.function(inputs=[theano.Param(batch_x),
                                        theano.Param(batch_y)],
                                outputs=self.errors,
                                givens={self.x: batch_x, self.y: batch_y})

        def scoref():
            """ returned function that scans the entire set given as input """
            return [score(batch_x, batch_y) for batch_x, batch_y in given_set]

        return scoref


class RegularizedNet(NeuralNet):
    """ Neural net with L1 and L2 regularization """
    def __init__(self, numpy_rng, theano_rng=None,
                 n_ins=100,
                 layers_types=[ReLU, ReLU, ReLU, LogisticRegression],
                 layers_sizes=[1024, 1024, 1024],
                 n_outs=2,
                 rho=0.9, eps=1.E-6,
                 L1_reg=0.,
                 L2_reg=0.,
                 debugprint=False):
        """
        TODO
        """
        super(RegularizedNet, self).__init__(numpy_rng, theano_rng, n_ins,
                layers_types, layers_sizes, n_outs, rho, eps, debugprint)

        L1 = shared(0.)
        for param in self.params:
            L1 += T.sum(abs(param))
        if L1_reg > 0.:
            self.cost = self.cost + L1_reg * L1
        L2 = shared(0.)
        for param in self.params:
            L2 += T.sum(param ** 2)
        if L2_reg > 0.:
            self.cost = self.cost + L2_reg * L2


class DropoutNet(NeuralNet):
    """ Neural net with dropout (see Hinton's et al. paper) """
    def __init__(self, numpy_rng, theano_rng=None,
                 n_ins=40*3,
                 layers_types=[Linear, ReLU, ReLU, ReLU, LogisticRegression],
                 layers_sizes=[1024, 1024, 1024, 1024],
                 dropout_rates=[0.2, 0.5, 0.5, 0.5, 0.5],
                 n_outs=62 * 3,
                 rho=0.9, eps=1.E-6,
                 debugprint=False):
        """
        TODO
        """
        super(DropoutNet, self).__init__(numpy_rng, theano_rng, n_ins,
                layers_types, layers_sizes, n_outs, rho, eps, debugprint)

        self.dropout_rates = dropout_rates
        dropout_layer_input = dropout(numpy_rng, self.x, p=dropout_rates[0])
        self.dropout_layers = []

        for layer, layer_type, n_in, n_out, dr in zip(self.layers,
                layers_types, self.layers_ins, self.layers_outs,
                dropout_rates[1:] + [0]):  # !!! we do not dropout anything
                                            # from the last layer !!!
            this_layer = layer_type(rng=numpy_rng,
                    input=dropout_layer_input, n_in=n_in, n_out=n_out,
                    W=layer.W * 1. / (1. - dr), # experimental
                    b=layer.b * 1. / (1. - dr)) # TODO check
            assert hasattr(this_layer, 'output')
            # N.B. dropout with dr==1 does not dropanything!!
            this_layer.output = dropout(numpy_rng, this_layer.output, dr)
            self.dropout_layers.append(this_layer)
            dropout_layer_input = this_layer.output

        assert hasattr(self.layers[-1], 'training_cost')
        assert hasattr(self.layers[-1], 'errors')
        # TODO standardize cost
        # these are the dropout costs
        self.mean_cost = self.dropout_layers[-1].negative_log_likelihood(self.y)
        self.cost = self.dropout_layers[-1].training_cost(self.y)

        # these is the non-dropout errors
        self.errors = self.layers[-1].errors(self.y)

    def __repr__(self):
        return super(DropoutNet, self).__repr__() + "\n"\
                + "dropout rates: " + str(self.dropout_rates)


def add_fit_and_score(class_to_chg):
    """ Mutates a class to add the fit() and score() functions to a NeuralNet.
    """
    from types import MethodType
    def fit(self, x_train, y_train, x_dev=None, y_dev=None,
            max_epochs=20, early_stopping=True, split_ratio=0.1, # TODO 100+ epochs
            method='adadelta', verbose=False, plot=False):
        """
        TODO
        """
        import time, copy
        if x_dev == None or y_dev == None:
            from sklearn.cross_validation import train_test_split
            x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train,
                    test_size=split_ratio, random_state=42)
        if method == 'sgd':
            train_fn = self.get_SGD_trainer()
        elif method == 'adagrad':
            train_fn = self.get_adagrad_trainer()
        elif method == 'adadelta':
            train_fn = self.get_adadelta_trainer()
        elif method == 'sag':
            #train_fn = self.get_SAG_trainer(R=1+numpy.max(numpy.sum(x_train**2, axis=1)))
            train_fn = self.get_SAG_trainer(R=numpy.max(numpy.sum(x_train**2, axis=1)))
        train_set_iterator = DatasetMiniBatchIterator(x_train, y_train)
        if method == 'sag':
            sag_train_set_iterator = DatasetMiniBatchIterator(x_train, y_train, randomize=True)
        dev_set_iterator = DatasetMiniBatchIterator(x_dev, y_dev)
        train_scoref = self.score_classif(train_set_iterator)
        dev_scoref = self.score_classif(dev_set_iterator)
        best_dev_loss = numpy.inf
        epoch = 0
        # TODO early stopping (not just cross val, also stop training)
        if plot:
            verbose = True
            self._costs = []
            self._train_errors = []
            self._dev_errors = []
            self._updates = []

        seen = numpy.zeros((x_train.shape[0] / BATCH_SIZE + 1,), dtype=numpy.bool)
        n_seen = 0

        while epoch < max_epochs:
            if not verbose:
                sys.stdout.write("\r%0.2f%%" % (epoch * 100./ max_epochs))
                sys.stdout.flush()
            avg_costs = []
            timer = time.time()
            if method == 'sag':
                for ind_minibatch, x, y in sag_train_set_iterator:
                    if not seen[ind_minibatch]:
                        seen[ind_minibatch] = 1
                        n_seen += 1
                    avg_cost = train_fn(x, y, ind_minibatch, n_seen)
                    if type(avg_cost) == list:
                        avg_costs.append(avg_cost[0])
                    else:
                        avg_costs.append(avg_cost)
            else:
                for x, y in train_set_iterator:
                    if method == 'sgd' or method == 'adagrad':
                        avg_cost = train_fn(x, y, lr=1.E-2)
                    elif method == 'adadelta':
                        avg_cost = train_fn(x, y)
                    if type(avg_cost) == list:
                        avg_costs.append(avg_cost[0])
                    else:
                        avg_costs.append(avg_cost)
            if verbose:
                mean_costs = numpy.mean(avg_costs)
                mean_train_errors = numpy.mean(train_scoref())
                print('  epoch %i took %f seconds' %
                      (epoch, time.time() - timer))
                print('  epoch %i, avg costs %f' %
                      (epoch, mean_costs))
                print('  epoch %i, training error %f' %
                      (epoch, mean_train_errors))
                if plot:
                    self._costs.append(mean_costs)
                    self._train_errors.append(mean_train_errors)
            dev_errors = numpy.mean(dev_scoref())
            if plot:
                self._dev_errors.append(dev_errors)
            if dev_errors < best_dev_loss:
                best_dev_loss = dev_errors
                best_params = copy.deepcopy(self.params)
                if verbose:
                    print('!!!  epoch %i, validation error of best model %f' %
                          (epoch, dev_errors))
            epoch += 1
        if not verbose:
            print("")
        for i, param in enumerate(best_params):
            self.params[i] = param

    def score(self, x, y):
        """ error rates """
        iterator = DatasetMiniBatchIterator(x, y)
        scoref = self.score_classif(iterator)
        return numpy.mean(scoref())

    class_to_chg.fit = MethodType(fit, None, class_to_chg)
    class_to_chg.score = MethodType(score, None, class_to_chg)


if __name__ == "__main__":
    add_fit_and_score(DropoutNet)
    add_fit_and_score(RegularizedNet)

    def nudge_dataset(X, Y):
        """
        This produces a dataset 5 times bigger than the original one,
        by moving the 8x8 images in X around by 1px to left, right, down, up
        """
        from scipy.ndimage import convolve
        direction_vectors = [
            [[0, 1, 0],
             [0, 0, 0],
             [0, 0, 0]],
            [[0, 0, 0],
             [1, 0, 0],
             [0, 0, 0]],
            [[0, 0, 0],
             [0, 0, 1],
             [0, 0, 0]],
            [[0, 0, 0],
             [0, 0, 0],
             [0, 1, 0]]]
        shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                      weights=w).ravel()
        X = numpy.concatenate([X] +
                              [numpy.apply_along_axis(shift, 1, X, vector)
                                  for vector in direction_vectors])
        Y = numpy.concatenate([Y for _ in range(5)], axis=0)
        return X, Y

    from sklearn import datasets, svm, naive_bayes
    from sklearn import cross_validation, preprocessing
    DIGITS = True
    FACES = True
    TWENTYNEWSGROUPS = False
    VERBOSE = True
    SCALE = True
    PLOT = True

    def train_models(x_train, y_train, x_test, y_test, n_features, n_outs,
            use_dropout=True, n_epochs=40, numpy_rng=None,
            svms=False, nb=False, deepnn=True):
        if svms:
            print("Linear SVM")
            classifier = svm.SVC(gamma=0.001)
            print(classifier)
            classifier.fit(x_train, y_train)
            print("score: %f" % classifier.score(x_test, y_test))

            print("RBF-kernel SVM")
            classifier = svm.SVC(kernel='rbf', class_weight='auto')
            print(classifier)
            classifier.fit(x_train, y_train)
            print("score: %f" % classifier.score(x_test, y_test))

        if nb:
            print("Multinomial Naive Bayes")
            classifier = naive_bayes.MultinomialNB()
            print(classifier)
            classifier.fit(x_train, y_train)
            print("score: %f" % classifier.score(x_test, y_test))

        if deepnn:
            import warnings
            warnings.filterwarnings("ignore")  # TODO remove

            if use_dropout:
                #n_epochs *= 4  TODO
                pass

            def new_dnn(dropout=False):
                if dropout:
                    print("Dropout DNN")
                    return DropoutNet(numpy_rng=numpy_rng, n_ins=n_features,
                        #layers_types=[LogisticRegression],
                        #layers_sizes=[],
                        #dropout_rates=[0.],
                        layers_types=[ReLU, ReLU, ReLU, LogisticRegression],
                        layers_sizes=[1000, 1000, 1000],
                        dropout_rates=[0., 0.5, 0.5, 0.5],
                        #layers_types=[ReLU, ReLU, LogisticRegression],
                        #layers_sizes=[200, 200],
                        #dropout_rates=[0., 0.5, 0.5],
                        n_outs=n_outs,
                        debugprint=0)
                else:
                    print("Simple (regularized) DNN")
                    return RegularizedNet(numpy_rng=numpy_rng, n_ins=n_features,
                        layers_types=[ReLU, ReLU, ReLU, LogisticRegression],
                        layers_sizes=[1000, 1000, 1000],
                        #layers_types=[ReLU, LogisticRegression],
                        #layers_sizes=[200],
                        n_outs=n_outs,
                        L1_reg=0.001/x_train.shape[0],
                        L2_reg=0.001/x_train.shape[0],
                        debugprint=0)

            import matplotlib.pyplot as plt
            plt.figure()
            ax1 = plt.subplot(221)
            ax2 = plt.subplot(222)
            ax3 = plt.subplot(223)
            ax4 = plt.subplot(224)  # TODO updates of the weights
            #for method in ['sag']:
            for method in ['sag', 'sgd', 'adagrad', 'adadelta']:
                dnn = new_dnn(use_dropout)
                print dnn
                dnn.fit(x_train, y_train, max_epochs=n_epochs, method=method, verbose=VERBOSE, plot=PLOT)
                test_error = dnn.score(x_test, y_test)
                print("score: %f" % (1. - test_error))
                ax1.plot(numpy.log10(dnn._costs), label=method)
                ax2.plot(numpy.log10(dnn._train_errors), label=method)
                ax3.plot(numpy.log10(dnn._dev_errors), label=method)
                #ax4.plot(dnn._updates, label=method) TODO
                ax4.plot([test_error for _ in range(10)], label=method)
            ax1.set_xlabel('epoch')
            ax1.set_ylabel('cost')
            ax2.set_xlabel('epoch')
            ax2.set_ylabel('train error')
            ax3.set_xlabel('epoch')
            ax3.set_ylabel('dev error')
            ax4.set_ylabel('test error')
            plt.legend()
            plt.savefig('training.png')


    if DIGITS:
        digits = datasets.load_digits()
        data = numpy.asarray(digits.data, dtype='float32')
        target = numpy.asarray(digits.target, dtype='int32')
        nudged_x, nudged_y = nudge_dataset(data, target)
        if SCALE:
            nudged_x = preprocessing.scale(nudged_x)
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(
                nudged_x, nudged_y, test_size=0.3, random_state=42)
        train_models(x_train, y_train, x_test, y_test, nudged_x.shape[1],
                     len(set(target)), numpy_rng=numpy.random.RandomState(123))

    if FACES:
        import logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(message)s')
        lfw_people = datasets.fetch_lfw_people(min_faces_per_person=70,
                                               resize=0.4)
        X = numpy.asarray(lfw_people.data, dtype='float32')
        if SCALE:
            X = preprocessing.scale(X)
        y = numpy.asarray(lfw_people.target, dtype='int32')
        target_names = lfw_people.target_names
        print("Total dataset size:")
        print("n samples: %d" % X.shape[0])
        print("n features: %d" % X.shape[1])
        print("n classes: %d" % target_names.shape[0])
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(
                    X, y, test_size=0.25, random_state=42)

        train_models(x_train, y_train, x_test, y_test, X.shape[1],
                     len(set(y)), numpy_rng=numpy.random.RandomState(123))

    if TWENTYNEWSGROUPS:
        from sklearn.feature_extraction.text import TfidfVectorizer
        newsgroups_train = datasets.fetch_20newsgroups(subset='train')
        vectorizer = TfidfVectorizer(encoding='latin-1', max_features=10000)
        #vectorizer = HashingVectorizer(encoding='latin-1')
        x_train = vectorizer.fit_transform(newsgroups_train.data)
        x_train = numpy.asarray(x_train.todense(), dtype='float32')
        y_train = numpy.asarray(newsgroups_train.target, dtype='int32')
        newsgroups_test = datasets.fetch_20newsgroups(subset='test')
        x_test = vectorizer.transform(newsgroups_test.data)
        x_test = numpy.asarray(x_test.todense(), dtype='float32')
        y_test = numpy.asarray(newsgroups_test.target, dtype='int32')
        train_models(x_train, y_train, x_test, y_test, x_train.shape[1],
                     len(set(y_train)),
                     numpy_rng=numpy.random.RandomState(123),
                     svms=False, nb=True, deepnn=True)

