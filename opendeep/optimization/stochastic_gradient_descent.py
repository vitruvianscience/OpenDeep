'''
.. module:: stochastic_gradient_descent

Generic stochastic gradient descent optimization with momentum (Nesterov acceleration) and annealing.
This also serves as the base class for other learning rate update algorithms, such as ADADELTA or RMSProp.
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
import time
# third party libraries
import numpy
import numpy.random as random
from theano.compat.python2x import OrderedDict  # use this compatibility OrderedDict
import theano.compat.six as six
# internal references
from opendeep import function, trunc, sharedX
from opendeep.optimization.optimizer import Optimizer
from opendeep.utils.decay import get_decay_function
from opendeep.data.iterators.sequential import SequentialIterator
import opendeep.data.dataset as datasets
from opendeep.utils.misc import make_time_units_string, get_shared_values, set_shared_values, raise_to_list

log = logging.getLogger(__name__)

class SGD(Optimizer):
    '''
    Stochastic gradient descent for training a model - includes early stopping, momentum, and annealing
    '''

    # Default values to use for some training parameters
    defaults = {"n_epoch": 1000,
                 "batch_size": 100,
                 "minimum_batch_size": 1,
                 "save_frequency": 10,
                 "early_stop_threshold": .9995,
                 "early_stop_length": 30,
                 "learning_rate": 0.25,
                 "lr_decay": "exponential",
                 "lr_factor": .995,
                 "momentum": 0.5,
                 'momentum_decay': 'linear',
                 'momentum_factor': 0,
                 'nesterov_momentum': True,
                 'flag_para_load': False}

    def __init__(self, model, dataset, iterator_class=SequentialIterator, config=None, defaults=defaults, rng=None,
                 n_epoch=None, batch_size=None, minimum_batch_size=None, save_frequency=None,
                 early_stop_threshold=None, early_stop_length=None, learning_rate=None, lr_decay=None, lr_factor=None,
                 momentum=None, momentum_decay=None, momentum_factor=None, nesterov_momentum=None, flag_para_load=None):
        # superclass init
        super(SGD, self).__init__(config=config, defaults=defaults)
        # config and defaults are now combined in self.args! yay!

        self.model = model
        self.dataset = dataset
        self.iterator = iterator_class

        # Training epochs - how many times to iterate over the whole dataset
        self.n_epoch = n_epoch or self.args.get('n_epoch')

        # Dataset iteration batch sizes - number of examples in each calculation
        self.batch_size         = batch_size or self.args.get('batch_size')
        self.minimum_batch_size = minimum_batch_size or self.args.get('minimum_batch_size')

        # Number of epochs between saving model parameters
        self.save_frequency = save_frequency or self.args.get('save_frequency')

        # Early stopping threshold and patience - by how much does the cost have to improve over a number of epochs
        self.early_stop_threshold = early_stop_threshold or self.args.get('early_stop_threshold')
        self.early_stop_length    = early_stop_length or self.args.get('early_stop_length')

        # Learning rate - how drastic of a step do the parameters change
        lr = learning_rate or self.args.get('learning_rate')
        self.learning_rate       = sharedX(lr, 'learning_rate')
        self.lr_scalers = self.model.get_lr_scalers()
        if lr_decay or self.args.get('lr_decay'):
            self.learning_rate_decay = get_decay_function(lr_decay or self.args.get('lr_decay'),
                                                          self.learning_rate,
                                                          self.learning_rate.get_value(),
                                                          lr_factor or self.args.get('lr_factor'))

        # Momentum - smoothing over the parameter changes (see Hinton)
        self.momentum = sharedX(momentum or self.args.get('momentum'), 'momentum')
        if self.args.get('momentum_decay'):
            self.momentum_decay = get_decay_function(momentum_decay or self.args.get('momentum_decay'),
                                                     self.momentum,
                                                     self.momentum.get_value(),
                                                     momentum_factor or self.args.get('momentum_factor'))
        self.nesterov_momentum = nesterov_momentum or self.args.get('nesterov_momentum')

        # RNG for working on random iterator
        if rng is None:
            random.seed(123)
            self.rng = random
        else:
            self.rng = rng
        # grab the parameters to use during training
        self.params = self.model.get_params()
        log.info("%s params: %s", str(type(self.model)), str(self.params))

        # Now time to create the training cost functions for the model - make sure to handle the possible
        # list of costs used for pretraining of certain parts of the model.
        train_costs = raise_to_list(self.model.get_train_cost())
        self.train_functions = []
        for i, train_cost in enumerate(train_costs):
            # Now create the training cost function for the model to use while training - update parameters
            # gradient!
            gradients, _ = self.model.get_gradient(cost=train_cost)

            # Calculate the optimizer updates each run
            # This is where the magic happens for a lot of sub-implementations of SGD, including AdaDelta!
            # It tells how to update the params each training epoch
            gradient_updates = self.get_updates(gradients)

            # Combine the updates from the model also if applicable
            train_updates = self.model.get_updates()
            if train_updates:
                train_updates.update(gradient_updates)
            else:
                train_updates = gradient_updates

            # Compile the training function!
            log.info('Compiling f_learn %d/%d function for model %s...', i+1, len(train_costs), str(type(self.model)))
            t = time.time()
            f_learn = function(inputs  = model.get_inputs()+model.get_targets(),
                               updates = train_updates,
                               outputs = train_cost,
                               name    = 'f_learn_%d' % i)
            log.info('f_learn compilation took %s', make_time_units_string(time.time() - t))
            self.train_functions.append(f_learn)

        # grab the expression(s) to use to monitor different model values during training
        log.debug("Compiling monitor functions...")
        monitor_t = time.time()
        monitors_expressions = self.model.get_monitors()
        self.monitors = {key: [] for key in monitors_expressions.keys()}
        for monitor_name, monitor_expr in monitors_expressions.iteritems():
            monitor_function = function(inputs=self.model.get_inputs()+self.model.get_targets(),
                                        updates=self.model.get_updates(),
                                        outputs=monitor_expr,
                                        name=monitor_name)
            self.monitors[monitor_name].append(monitor_function)
        log.debug("Compilation done. Took %s", make_time_units_string(time.time() - monitor_t))


    def get_updates(self, grads):
        """
        From Pylearn2 (https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/training_algorithms/learning_rule.py)

        Implements momentum as described in Section 9 of
        "A Practical Guide to Training Restricted Boltzmann Machines",
        Geoffrey Hinton.
        Parameters are updated by the formula:
        inc := momentum * inc - learning_rate * d cost / d param
        param := param + inc

        Also has the option to implement Nesterov momentum (accelerated momentum), which works better in a lot of cases.

        :param grads: OrderedDict
        An OrderedDict of (parameter, gradient) for the model's gradients
        :return: OrderedDict
        Updates at each training step
        """
        log.debug('Setting up Stochastic Gradient Descent with momentum for optimizer...')
        updates = OrderedDict()
        for (param, gradient) in six.iteritems(grads):
            vel = sharedX(param.get_value() * 0.)
            assert param.dtype == vel.dtype
            assert gradient.dtype == param.dtype
            if param.name is not None:
                vel.name = 'vel_' + param.name

            scaled_lr = self.learning_rate * self.lr_scalers.get(param, 1.)
            updates[vel] = self.momentum * vel - scaled_lr * gradient

            inc = updates[vel]
            if self.nesterov_momentum:
                log.debug('Using Nesterov momentum')
                inc = self.momentum * inc - scaled_lr * gradient

            assert inc.dtype == vel.dtype
            updates[param] = param + inc

        return updates


    def train(self, continue_training=False):
        # make sure to deal with a list of train_cost functions - for layer-wise pretraining!
        # this list of training functions was created during __init__()
        start_time = time.time()
        for func_i, train_function in enumerate(self.train_functions):
            # Determine if this function is unsupervised or not by looking at the number of inputs
            # to the f_learn function.
            # If there is only one input, it is unsupervised, otherwise, it is supervised.
            # This workaround was provided by Pascal Lamblin on the theano-users google group
            num_inputs = len([i for i in train_function.maker.inputs if not i.shared])
            if num_inputs == 1:
                log.debug("Model is unsupervised: 1 input to f_learn.")
                unsupervised = True
            elif num_inputs == 2:
                log.debug("Model is supervised: 2 inputs to f_learn.")
                unsupervised = False
            else:
                log.error(
                    "Number of inputs to f_learn on model %s was %s. "
                    "Needs to be 1 for unsupervised or 2 for supervised.",
                    str(type(self.model)),
                    str(num_inputs)
                )
                raise AssertionError(
                    "Number of inputs to f_learn on model %s was %s. "
                    "Needs to be 1 for unsupervised or 2 for supervised." %
                    str(type(self.model)),
                    str(num_inputs)
                )

            log.info("-----------TRAINING %s function %d/%d FOR %d EPOCHS (continue_training=%s)-----------",
                     str(type(self.model)), func_i+1, len(self.train_functions), self.n_epoch, str(continue_training))

            log.debug("Train dataset size is: %s", self.dataset.getDataShape(datasets.TRAIN))
            if self.dataset.hasSubset(datasets.VALID):
                log.debug("Valid dataset size is: %s", self.dataset.getDataShape(datasets.VALID))
            if self.dataset.hasSubset(datasets.TEST):
                log.debug("Test dataset size is: %s", self.dataset.getDataShape(datasets.TEST))

            self.STOP    = False
            self.epoch_counter = 0
            if not continue_training:
                # reset the learning rate
                if hasattr(self, 'learning_rate_decay'):
                    self.learning_rate_decay.reset()
                # reset the other model decaying functions
                for decay_param in self.model.get_decay_params():
                    decay_param.reset()

            self.times       = []
            self.best_cost   = float('inf')
            self.best_params = None
            self.patience    = 0

            t = time.time()

            while not self.STOP:
                try:
                    self.STOP = self._perform_one_epoch(train_function, unsupervised)
                except KeyboardInterrupt:
                    log.info("STOPPING EARLY FROM KEYBOARDINTERRUPT")
                    self.STOP = True

            #save params
            if self.best_params is not None:
                log.debug("Restoring best model parameters...")
                set_shared_values(self.params, self.best_params)
            log.debug("Saving model parameters...")
            self.model.save_params('trained_epoch_'+str(self.epoch_counter)+'.pkl')

            log.info("------------TRAIN TIME TOOK %s---------", make_time_units_string(time.time()-t))

        log.info("------------TOTAL %s TRAIN TIME TOOK %s---------",
                 str(type(self.model)), make_time_units_string(time.time() - start_time))


    def _perform_one_epoch(self, f_learn, unsupervised):
            self.epoch_counter += 1
            t = time.time()
            log.info('EPOCH %s', str(self.epoch_counter))

            #train
            train_costs = []
            train_monitors = {key: [] for key in self.monitors.keys()}
            for x, y in self.iterator(self.dataset, datasets.TRAIN, self.batch_size, self.minimum_batch_size, self.rng):
                if unsupervised:
                    train_costs.append(f_learn(x))
                    self.call_monitors(monitors_dict=train_monitors, inputs=[x])
                else:
                    train_costs.append(f_learn(x, y))
                    self.call_monitors(monitors_dict=train_monitors, inputs=[x, y])

            log.info('Train cost: %s', trunc(numpy.mean(train_costs, 0)))
            if len(self.monitors.keys()) > 0:
                log.info('Train monitors: %s',
                         str({key: numpy.mean(value, 0) for key, value in train_monitors.items()}))

            #valid
            if self.dataset.hasSubset(datasets.VALID) and len(self.monitors.keys()) > 0:
                valid_monitors = {key: [] for key in self.monitors.keys()}
                for x, y in self.iterator(
                        self.dataset, datasets.VALID, self.batch_size, self.minimum_batch_size, self.rng
                ):
                    if unsupervised:
                        self.call_monitors(monitors_dict=valid_monitors, inputs=[x])
                    else:
                        self.call_monitors(monitors_dict=valid_monitors, inputs=[x, y])

                log.info('Valid monitors: %s',
                         str({key: numpy.mean(value, 0) for key, value in valid_monitors.items()}))

            #test
            if self.dataset.hasSubset(datasets.TEST) and len(self.monitors.keys()) > 0:
                test_monitors = {key: [] for key in self.monitors.keys()}
                for x, y in self.iterator(
                        self.dataset, datasets.TEST, self.batch_size, self.minimum_batch_size, self.rng
                ):
                    if unsupervised:
                        self.call_monitors(monitors_dict=test_monitors, inputs=[x])
                    else:
                        self.call_monitors(monitors_dict=test_monitors, inputs=[x, y])

                log.info('Test monitors: %s', str({key: numpy.mean(value, 0) for key, value in test_monitors.items()}))

            # check for early stopping on train costs
            cost = numpy.sum(train_costs)
            if cost < self.best_cost*self.early_stop_threshold:
                self.patience = 0
                self.best_cost = cost
                # save the parameters that made it the best
                self.best_params = get_shared_values(self.params)
            else:
                self.patience += 1

            # check for stopping either from n_epochs or from threshold/patience
            stop = False
            if self.epoch_counter >= self.n_epoch:
                log.info("Stopping (reached max number of epochs)...")
                stop = True
            if self.patience >= self.early_stop_length:
                log.info("Stopping early (reached stop threshold)...")
                stop = True

            timing = time.time() - t
            self.times.append(timing)

            log.info('time: ' + make_time_units_string(timing))

            log.info('remaining time: ' +
                     make_time_units_string((self.n_epoch - self.epoch_counter) * numpy.mean(self.times)))

            if (self.epoch_counter % self.save_frequency) == 0:
                #save params
                self.model.save_params('trained_epoch_'+str(self.epoch_counter)+'.pkl')

            # ANNEAL!
            if not stop:
                if hasattr(self, 'learning_rate_decay'):
                    self.learning_rate_decay.decay()
                if hasattr(self, 'momentum_decay'):
                    self.momentum_decay.decay()
                for decay_param in self.model.get_decay_params():
                    decay_param.decay()

            # return whether or not to stop this epoch
            return stop

    def call_monitors(self, monitors_dict, inputs):
        for key in self.monitors.keys():
            monitor_functions = raise_to_list(self.monitors[key])
            for monitor_function in monitor_functions:
                monitors_dict[key].append(monitor_function(*inputs))