"""
Basic interface for an optimizer - a training algorithm for models.

Some information from Andrej Karpathy:
'In my own experience, Adagrad/Adadelta are "safer" because they don't depend so strongly on setting of learning rates
(with Adadelta being slightly better), but well-tuned SGD+Momentum almost always converges faster and at better final
values.' http://cs.stanford.edu/people/karpathy/convnetjs/demo/trainers.html

Also see:
'Practical recommendations for gradient-based training of deep architectures'
Yoshua Bengio
http://arxiv.org/abs/1206.5533

'No More Pesky Learning Rates'
Tom Schaul, Sixin Zhang, Yann LeCun
http://arxiv.org/abs/1206.1106

Attributes
----------
TRAIN_COST_KEY : str
    The monitor name to use for the training cost. (Optimizer will always automatically monitor the training cost).
"""
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
import time
import os
# third party
import numpy
import theano.tensor as T
from theano.compat.python2x import OrderedDict
from theano.compat import six
# internal references
from opendeep import sharedX, function, trunc
from opendeep.data.dataset import Dataset, TRAIN, VALID, TEST, get_subset_strings
from opendeep.models.model import Model
from opendeep.monitor.monitor import collapse_channels
from opendeep.monitor.out_service import FileService
from opendeep.utils.decay import get_decay_function
from opendeep.utils.misc import raise_to_list, make_time_units_string, get_shared_values, set_shared_values, add_kwargs_to_dict

log = logging.getLogger(__name__)

TRAIN_COST_KEY = 'train_cost'


class Optimizer(object):
    """
    Default interface for an optimizer implementation - this provides the necessary parameter updates when
    training a model on a dataset using an online stochastic process.
    """
    def __init__(self, model, dataset,
                 n_epoch=1000, batch_size=100, minimum_batch_size=1,
                 save_frequency=10, early_stop_threshold=.9995, early_stop_length=30,
                 learning_rate=1e-3, lr_decay='exponential', lr_factor=1,
                 **kwargs):
        """
        Initialize the Optimizer.

        Parameters
        ----------
        model : Model
            The Model to train.
        dataset : Dataset
            The Dataset to use when training the Model.
        n_epoch : int
            how many training iterations over the dataset to go.
        batch_size : int
            How many examples from the training dataset to use in parallel.
        minimum_batch_size : int
            The minimum number of examples required at a time (for things like time series, this would be > 1).
        save_frequency : int
            How many epochs to train between each new save of the Model's parameters.
        early_stop_threshold : float
            The factor by how much the best validation training score needs to improve to determine early stopping.
        early_stop_length : int
            The patience or number of epochs to wait after the early_stop_threshold has been reached before stopping.
        learning_rate : float
            The multiplicative amount to adjust parameters based on their gradient values.
        lr_decay : str
            The type of decay function to use for changing the learning rate over epochs. See
            `opendeep.utils.decay` for options.
        lr_factor : float
            The amount to use for the decay function when changing the learning rate over epochs. See
            `opendeep.utils.decay` for its effect for given decay functions.
        """
        log.info("Initializing optimizer %s", str(type(self)))

        if early_stop_threshold is None:
            early_stop_threshold = 1.
        if save_frequency is None:
            save_frequency = 1000000
        if early_stop_length is None:
            early_stop_length = 100

        self.args = locals().copy()
        self.args.pop('self')
        kwargs = self.args.pop('kwargs')
        self.args = add_kwargs_to_dict(kwargs, self.args)
        # log the arguments
        log.info("optimizer config args: %s", str(self.args))

        assert isinstance(model, Model), "Optimizer input model needs to be an opendeep Model class!"
        assert isinstance(dataset, Dataset), "Optimizer input dataset needs to be an opendeep Dataset class!"
        self.model = model
        self.dataset = dataset

        # Learning rate - how drastic of a step do the parameters change
        self.learning_rate = sharedX(learning_rate, 'learning_rate')
        self.lr_scalers = self.model.get_lr_scalers()
        if lr_decay:
            self.learning_rate_decay = get_decay_function(lr_decay,
                                                          self.learning_rate,
                                                          self.learning_rate.get_value(),
                                                          lr_factor)
        else:
            self.learning_rate_decay = False

        self.noise_switches = raise_to_list(self.model.get_noise_switch())
        self.batch_size = batch_size
        self.minimum_batch_size = minimum_batch_size
        self.n_epoch = n_epoch
        self.save_frequency = save_frequency
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_length = early_stop_length

    def _get_batch_indices(self, data_lengths):
        """
        Computes the tuples of (start_index, end_index) that represent the appropriate slices of the concatenated
        dataset with regards to the given data_lengths. This allows for lists of data lengths to represent sequences,
        so that the concatenated batches returned do not overstep the start of a new sequence.

        Parameters
        ----------
        data_lengths : list(int) or int
            List of num_examples for each dataset (the length of the datasets - this is a list in the case of
            sequences).

        Returns
        -------
        list((int, int))
            List of tuples (start, end) representing the batch slices for the total dataset if it were concatenated.
        """
        batch_indices = []
        start_idx = 0
        for len in raise_to_list(data_lengths):
            # integer division to determine number of whole batches for this length
            n_batches = len / int(self.batch_size)
            # add the (start_idx, end_idx) tuple to the list
            for i in range(n_batches):
                end_idx = start_idx + self.batch_size
                batch_indices.append((start_idx, end_idx))
                start_idx = end_idx
            # remainder to find number of leftover examples
            remainder = numpy.remainder(len, self.batch_size)
            end_idx = start_idx + remainder
            # check if it is bigger than the minimum allowed size
            if remainder >= self.minimum_batch_size:
                batch_indices.append((start_idx, end_idx))
            start_idx = end_idx
        return batch_indices

    def _get_givens_subset(self, subset, batch_slice):
        """
        This translates a batch slice of start and end indices into the actual data from the given subset.

        Parameters
        ----------
        subset : int
            The subset to use - determined in opendeep.data.datasets as TRAIN, VALID, or TEST attributes.
        batch_slice : symbolic slice
            The symbolic slice to grab from the data.

        Returns
        -------
        OrderedDict
            The givens to provide to a function where it sets the input variable to the actual batch representation
            of data from the dataset: (input_variable: data[batch])
        """
        # translate the data_idx into the givens for the model
        # first get the lists of input variables the model requires - inputs and targets
        model_inputs = raise_to_list(self.model.get_inputs())
        model_targets = raise_to_list(self.model.get_targets())
        givens = None
        if self.dataset.getSubset(subset)[0] is not None:
            # grab the data and labels
            data, labels = self.dataset.getSubset(subset)
            # create the givens for the input function as pairs of (input_variable: sliced_data)
            givens = OrderedDict(zip(model_inputs, [data[batch_slice]]))
            # include labels as well if they are required by the model
            if model_targets is not None and len(model_targets) > 0:
                if labels is None:
                    log.error("No labels in the dataset!")
                    raise AssertionError, "No lables in the dataset!"
                givens.update(OrderedDict(zip(model_targets, [labels[batch_slice]])))
        else:
            log.warning("Dataset doesn't have subset %s" % get_subset_strings(subset))

        return givens

    def get_updates(self, gradients):
        """
        This returns the parameter updates to use during training. It defaults to only using (annealed) learning rate.

        Parameters
        ----------
        gradients : dict
            A dictionary mapping from the model's parameters to their
            gradients.

        Returns
        -------
        updates : OrderdDict
            A dictionary mapping from the old model parameters, to their new
            values after a single iteration of the learning rule.
        """
        log.debug('Setting up Stochastic Gradient Descent for optimizer...')
        updates = OrderedDict()
        for (param, gradient) in six.iteritems(gradients):
            scaled_lr = self.learning_rate * self.lr_scalers.get(param, 1.)
            updates[param] = param - scaled_lr * gradient
        return updates

    def get_lr_monitor(self):
        """
        Returns a monitor dictionary to the Optimizer's learning rate.

        Returns
        -------
        dict
            Mapping 'learning_rate' to `self.learning_rate` shared variable.
        """
        return {'learning_rate': self.learning_rate}

    def train(self, monitor_channels=None, train_outservice=None, plot=None, continue_training=False):
        """
        This method performs the training!!!
        It is an online training method that goes over minibatches from the dataset for a number of epochs,
        updating parameters after each minibatch.

        You can disrupt training with a KeyBoardInterrupt and it should exit/save parameters gracefully.

        Parameters
        ----------
        monitor_channels : list(MonitorsChannel or Monitor), optional
            The list of channels or monitors containing monitor expressions/variables to compile and evaluate
            on the data.
        train_outservice : OutService, optional
            The OutService to use for the automatically created train_cost monitor. Default of None just outputs
            to logs.
        plot : Plot, optional
            The Plot object to use if we want to graph the outputs (uses bokeh server).
        continue_training : bool
            Whether to continue training from a previous point.
        """
        ###############################################
        # theano index variable to use on the dataset #
        ###############################################
        # index to a [mini]batch - both start and end
        data_idx = T.iscalar('data_index')
        data_end_idx = T.iscalar('data_end_index')
        function_input = [data_idx, data_end_idx]
        batch_slice = slice(data_idx, data_end_idx)

        # compute number of minibatches for training, validation and testing
        # shapes is list of list - input list of datasets to optimizer (for multiple inputs), and each dataset
        # could be a list of shared variables (like multiple sequences from files)
        train_data_shapes = raise_to_list(self.dataset.getDataShape(TRAIN))
        valid_data_shapes = raise_to_list(self.dataset.getDataShape(VALID))
        test_data_shapes = raise_to_list(self.dataset.getDataShape(TEST))

        # train_batches is going to be lists of tuples that contain the start and end indices for train data.
        # this is more useful in the case of datasets that are lists of sequences, so that the start and end
        # indices can make sure a batch does not cross the sequence boundary on the concatenated data
        train_data_lens = [shape[0] for shape in train_data_shapes]
        self.train_batches = self._get_batch_indices(train_data_lens)

        if valid_data_shapes is not None:
            valid_data_lens = [shape[0] for shape in valid_data_shapes]
            self.valid_batches = self._get_batch_indices(valid_data_lens)
        else:
            self.valid_batches = None
        if test_data_shapes is not None:
            test_data_lens = [shape[0] for shape in test_data_shapes]
            self.test_batches = self._get_batch_indices(test_data_lens)
        else:
            self.test_batches = None

        # create the givens for the input function as pairs of (input_variable: sliced_data)
        train_givens = self._get_givens_subset(TRAIN, batch_slice)
        valid_givens = self._get_givens_subset(VALID, batch_slice)
        test_givens = self._get_givens_subset(TEST, batch_slice)

        # Now time to create the gradient updates for the model - make sure to handle the possible
        # list of costs used for pretraining of certain parts of the model.
        train_costs = raise_to_list(self.model.get_train_cost())
        train_updates = []
        self.gradients = []
        for i, train_cost in enumerate(train_costs):
            # Now create the training cost function for the model to use while training - update parameters
            # gradient!
            gradients, _ = self.model.get_gradient(cost=train_cost)
            self.gradients.append(gradients)

            # Calculate the optimizer updates each run
            # This is where the magic happens for a lot of sub-implementations of SGD!
            # It tells how to update the params each training epoch
            gradient_updates = self.get_updates(gradients)

            # Combine the updates from the model also if applicable
            updates = self.model.get_updates()
            if updates:
                updates.update(gradient_updates)
            else:
                updates = gradient_updates
            train_updates.append(updates)

        # grab the model parameters to use during training
        self.params = self.model.get_params()
        log.info("%s params: %s", str(type(self.model)), str(self.params))

        # deal with the monitor channels if they were given (or take them from the plot)
        if monitor_channels is None and plot is not None and len(plot.channels) > 0:
            monitor_channels = plot.channels
        self.train_monitors_dict = {}
        self.valid_monitors_dict = {}
        self.test_monitors_dict = {}
        self.train_monitors_outservice_dict = {}
        self.valid_monitors_outservice_dict = {}
        self.test_monitors_outservice_dict = {}
        if monitor_channels:
            # collapse the appropriate monitors into their (name, expression, out_service) tuples
            train_collapsed = collapse_channels(monitor_channels, train=True)
            valid_collapsed = collapse_channels(monitor_channels, valid=True)
            test_collapsed  = collapse_channels(monitor_channels, test=True)
            # get name: expression dictionary
            self.train_monitors_dict = OrderedDict([(name, expression) for name, expression, _ in train_collapsed])
            self.valid_monitors_dict = OrderedDict([(name, expression) for name, expression, _ in valid_collapsed])
            self.test_monitors_dict  = OrderedDict([(name, expression) for name, expression, _ in test_collapsed])
            # get name: outservice dictionary
            self.train_monitors_outservice_dict = OrderedDict([(name, out) for name, _, out in train_collapsed])
            self.valid_monitors_outservice_dict = OrderedDict([(name, out) for name, _, out in valid_collapsed])
            self.test_monitors_outservice_dict  = OrderedDict([(name, out) for name, _, out in test_collapsed])
        # finally deal with an outservice provided to monitor training cost
        self.train_outservice = train_outservice
        # remove redundant files made by the fileservice for the train monitor.
        # TODO: THIS FEELS LIKE A HACK. I don't like it.
        if isinstance(self.train_outservice, FileService):
            os.remove(self.train_outservice.valid_filename)
            os.remove(self.train_outservice.test_filename)

        #######################################
        # compile train and monitor functions #
        #######################################
        train_functions = []
        for i in range(len(train_costs)):
            updates = train_updates[i]
            train_cost = train_costs[i]
            # Compile the training function!
            log.info('Compiling f_learn %d/%d function for model %s...', i + 1, len(train_updates),
                     str(type(self.model)))
            t = time.time()

            f_learn = function(inputs=function_input,
                               updates=updates,
                               outputs=[train_cost] + self.train_monitors_dict.values(),
                               givens=train_givens,
                               name='f_learn_%d' % i)

            log.info('f_learn compilation took %s', make_time_units_string(time.time() - t))
            train_functions.append(f_learn)

        # figure out if we want valid and test
        self.valid_flag = (self.dataset.getSubset(VALID)[0] is not None) and (len(self.valid_monitors_dict) > 0)
        self.test_flag = (self.dataset.getSubset(TEST)[0] is not None) and (len(self.test_monitors_dict) > 0)
        # Now compile the monitor functions!
        log.debug("Compiling monitor functions...")
        monitor_t = time.time()
        # valid monitors
        if self.valid_flag:
            self.valid_monitor_function = function(
                inputs=function_input,
                updates=self.model.get_updates(),
                outputs=self.valid_monitors_dict.values(),
                givens=valid_givens,
                name='valid_monitor_function'
            )
        else:
            self.valid_monitor_function = None

        # test monitors
        if self.test_flag:
            self.test_monitor_function = function(
                inputs=function_input,
                updates=self.model.get_updates(),
                outputs=self.test_monitors_dict.values(),
                givens=test_givens,
                name='test_monitor_function'
            )
        else:
            self.test_monitor_function = None

        log.debug("Compilation done. Took %s", make_time_units_string(time.time() - monitor_t))

        ##################
        # start training #
        ##################
        # make sure to deal with a list of train_cost functions - for layer-wise pretraining!
        # this list of training functions was created during __init__()
        start_time = time.time()
        for func_i, train_function in enumerate(train_functions):
            log.info("-----------TRAINING %s function %d/%d FOR %d EPOCHS (continue_training=%s)-----------",
                     str(type(self.model)), func_i + 1, len(train_functions), self.n_epoch, str(continue_training))

            log.debug("Train dataset size is: %s", self.dataset.getDataShape(TRAIN))
            if self.dataset.getSubset(VALID)[0] is not None:
                log.debug("Valid dataset size is: %s", self.dataset.getDataShape(VALID))
            if self.dataset.getSubset(TEST)[0] is not None:
                log.debug("Test dataset size is: %s", self.dataset.getDataShape(TEST))

            self.STOP = False
            self.epoch_counter = 0
            if not continue_training:
                # reset any decay params
                for decay_param in self.get_decay_params():
                    decay_param.reset()

            self.times = []
            self.best_cost = numpy.inf
            self.best_params = None
            self.patience = 0

            t = time.time()

            while not self.STOP:
                try:
                    self.STOP = self._perform_one_epoch(train_function, plot)
                except KeyboardInterrupt:
                    log.info("STOPPING EARLY FROM KEYBOARDINTERRUPT")
                    self.STOP = True

            # save params
            if self.best_params is not None:
                log.debug("Restoring best model parameters...")
                set_shared_values(self.params, self.best_params)
            log.debug("Saving model parameters...")
            self.model.save_params('trained_epoch_' + str(self.epoch_counter) + '.pkl')

            log.info("------------TRAIN TIME TOOK %s---------", make_time_units_string(time.time() - t))

        log.info("------------TOTAL %s TRAIN TIME TOOK %s---------",
                 str(type(self.model)), make_time_units_string(time.time() - start_time))


    def _perform_one_epoch(self, f_learn, plot=None):
        """
        Performs a single training iteration with the given learn function.
        """
        self.epoch_counter += 1
        t = time.time()
        log.info('EPOCH %s', str(self.epoch_counter))

        # set the noise switches on for training function! (this is where things like dropout happen)
        switch_vals = []
        if len(self.noise_switches) > 0 and (self.valid_flag or self.test_flag or self.epoch_counter == 1):
            log.debug("Turning on %s noise switches", str(len(self.noise_switches)))
            switch_vals = [switch.get_value() for switch in self.noise_switches]
            [switch.set_value(1.) for switch in self.noise_switches]

        # train
        train_costs = []
        train_monitors = {key: [] for key in self.train_monitors_dict.keys()}
        for batch_start, batch_end in self.train_batches:
            _outs = raise_to_list(f_learn(batch_start, batch_end))
            train_costs.append(_outs[0])
            # handle any user defined monitors
            if len(train_monitors) > 0:
                current_monitors = zip(self.train_monitors_dict.keys(), _outs[1:])
                for name, val in current_monitors:
                    train_monitors[name].append(val)

        # get the mean values for the batches
        mean_train = numpy.mean(train_costs, 0)
        current_mean_monitors = {key: numpy.mean(vals, 0) for key, vals in train_monitors.items()}
        # log the mean values!
        log.info('Train cost: %s', trunc(mean_train))
        if len(current_mean_monitors) > 0:
            log.info('Train monitors: %s', str(current_mean_monitors))
        # send the values to their outservices
        if self.train_outservice:
            self.train_outservice.write(mean_train, TRAIN)
        for name, service in self.train_monitors_outservice_dict.items():
            if name in current_mean_monitors and service:
                service.write(current_mean_monitors[name], TRAIN)
        # if there is a plot, also send them over!
        if plot:
            current_mean_monitors.update({TRAIN_COST_KEY: mean_train})
            plot.update_plots(epoch=self.epoch_counter, monitors=current_mean_monitors)

        # set the noise switches off for valid and test sets! we assume unseen data is noisy anyway :)
        if len(self.noise_switches) > 0 and (self.valid_flag or self.test_flag):
            log.debug("Turning off %s noise switches", str(len(self.noise_switches)))
            [switch.set_value(0.) for switch in self.noise_switches]

        # valid
        if self.valid_flag:
            valid_monitors = {key: [] for key in self.valid_monitors_dict.keys()}
            for batch_start, batch_end in self.valid_batches:
                _outs = raise_to_list(self.valid_monitor_function(batch_start, batch_end))
                current_monitors = zip(self.valid_monitors_dict.keys(), _outs)
                for name, val in current_monitors:
                    valid_monitors[name].append(val)

            # get the mean values for the batches
            current_mean_monitors = {key: numpy.mean(vals, 0) for key, vals in valid_monitors.items()}
            # log the mean values!
            log.info('Valid monitors: %s', str(current_mean_monitors))
            # send the values to their outservices
            for name, service in self.valid_monitors_outservice_dict.items():
                if name in current_mean_monitors and service:
                    service.write(current_mean_monitors[name], VALID)
            # if there is a plot, also send them over!
            if plot:
                plot.update_plots(epoch=self.epoch_counter, monitors=current_mean_monitors)

        #test
        if self.test_flag:
            test_monitors = {key: [] for key in self.test_monitors_dict.keys()}
            for batch_start, batch_end in self.test_batches:
                _outs = raise_to_list(self.test_monitor_function(batch_start, batch_end))
                current_monitors = zip(self.test_monitors_dict.keys(), _outs)
                for name, val in current_monitors:
                    test_monitors[name].append(val)

            # get the mean values for the batches
            current_mean_monitors = {key: numpy.mean(vals, 0) for key, vals in test_monitors.items()}
            # log the mean values!
            log.info('Test monitors: %s', str(current_mean_monitors))
            # send the values to their outservices
            for name, service in self.test_monitors_outservice_dict.items():
                if name in current_mean_monitors and service:
                    service.write(current_mean_monitors[name], TEST)
            # if there is a plot, also send them over!
            if plot:
                plot.update_plots(epoch=self.epoch_counter, monitors=current_mean_monitors)

        # check for early stopping on train costs
        cost = numpy.sum(train_costs)
        if cost < self.best_cost * self.early_stop_threshold:
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

        log.debug('remaining time: ' +
                 make_time_units_string((self.n_epoch - self.epoch_counter) * numpy.mean(self.times)))

        if (self.epoch_counter % self.save_frequency) == 0:
            #save params
            self.model.save_params('trained_epoch_' + str(self.epoch_counter) + '.pkl')

        # ANNEAL!
        if not stop:
            # perform the appropriate decay on the decay functions/parameters for this optimizer and model
            for decay_param in self.get_decay_params():
                decay_param.decay()

        # reset the switches
        if len(self.noise_switches) > 0:
            [switch.set_value(val) for switch, val in zip(self.noise_switches, switch_vals)]

        # return whether or not to stop this epoch
        return stop

    def get_decay_params(self):
        """
        Returns a list of all the Decay objects to decay during training.

        Returns
        -------
        list
            List of Decay objects to use after each training epoch - in this case the
            learning rate decay.
        """
        decay_params = self.model.get_decay_params()
        if hasattr(self, 'learning_rate_decay') and self.learning_rate_decay:
            decay_params.append(self.learning_rate_decay)
        return decay_params