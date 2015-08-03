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
# standard libraries
import logging
import time
import os
import warnings
# third party
import numpy
import theano.tensor as T
from theano.compat.python2x import OrderedDict
from theano.compat import six
# internal references
from opendeep.utils.constructors import sharedX, function
from opendeep.data.dataset import Dataset
from opendeep.models.model import Model
from opendeep.monitor.monitor import collapse_channels
from opendeep.monitor.out_service import FileService
from opendeep.utils.decay import get_decay_function
from opendeep.utils.misc import raise_to_list, make_time_units_string, \
    get_shared_values, set_shared_values, add_kwargs_to_dict, trunc
from opendeep.utils.batch import minibatch
from opendeep.utils.misc import min_normalized_izip

log = logging.getLogger(__name__)

TRAIN_COST_KEY = 'train_cost'


class Optimizer(object):
    """
    Default interface for an optimizer implementation - this provides the necessary parameter updates when
    training a model on a dataset using an online stochastic process. The base framework for performing
    stochastic gradient descent.
    """
    def __init__(self, dataset, model=None,
                 epochs=1000, batch_size=100, min_batch_size=1,
                 save_freq=10, stop_threshold=None, stop_patience=50,
                 learning_rate=1e-3, lr_decay=None, lr_decay_factor=None,
                 grad_clip=None, hard_clip=False,
                 **kwargs):
        """
        Initialize the Optimizer.

        Parameters
        ----------
        dataset : Dataset
            The Dataset to use when training the Model.
        model : Model
            The Model to train. Needed if the Optimizer isn't being passed to a Model's .train() method.
        epochs : int
            How many training iterations over the dataset to go.
        batch_size : int
            How many examples from the training dataset to use in parallel.
        min_batch_size : int
            The minimum number of examples required at a time (for things like time series, this would be > 1).
        save_freq : int, optional
            How many epochs to train between each new save of the Model's parameters.
        stop_threshold : float, optional
            The factor by how much the best validation training score needs to improve to determine early stopping.
        stop_patience : int, optional
            The patience or number of epochs to wait after the stop_threshold has been reached before stopping.
        learning_rate : float
            The multiplicative amount to adjust parameters based on their gradient values.
        lr_decay : str
            The decay function to use for changing the learning rate over epochs. See
            `opendeep.utils.decay` for classes of decay and documentation.
        lr_decay_factor : float
            The amount of decay to use for the ``lr_decay`` type of decay.
        grad_clip : float, optional
            Whether to clip gradients. This will clip the norm of the gradients either with a hard cutoff or rescaling.
        hard_clip : bool
            Whether to use a hard cutoff or rescaling for clipping gradients.
        """
        log.info("Initializing optimizer %s", str(type(self)))

        # Deal with early stopping None initializations.
        if not stop_threshold:
            stop_threshold = numpy.inf
        if not save_freq:
            save_freq = 1000000
        if not stop_patience:
            stop_patience = 1

        # Put all init parameters in self.args so we can log the initial configuration.
        self.args = locals().copy()
        self.args.pop('self')
        kwargs = self.args.pop('kwargs')
        self.args = add_kwargs_to_dict(kwargs, self.args)
        # log the arguments
        log.info("Optimizer config args: %s", str(self.args))
        # if the optimizer wasn't initialized with a Model (train() being called from the model class itself),
        # just return. (This seems kinda hacky but hey, people wanted .train() to happen from Model and there
        # wasn't really a better way unless the epoch looping logic was in that method for Model. That wasn't
        # the best option because other methods besides stochastic ones can exist for optimizers in the future.
        # TODO: fix this up - feels like a hack just to make model.train() work...
        if not model:
            return
        # Otherwise, things are proceeding as normal. Carry on...

        assert isinstance(model, Model), "Optimizer input model needs to be a Model class! " \
                                         "Found %s" % str(type(model))
        assert isinstance(dataset, Dataset), "Optimizer input dataset needs to be a Dataset class! " \
                                             "Found %s" % str(type(dataset))

        n_model_inputs = len(raise_to_list(model.get_inputs()))
        n_model_targets = len(raise_to_list(model.get_targets()) or [])
        self.unsupervised = (n_model_targets is 0)
        # make sure the number of inputs/targets matches up with the dataset properties
        # train
        assert n_model_inputs == len(raise_to_list(dataset.train_inputs)), \
            "Dataset has %d train inputs, while model expects %d" % \
            (len(raise_to_list(dataset.train_inputs)), n_model_inputs)
        if not self.unsupervised:
            assert n_model_targets == len(raise_to_list(dataset.train_targets) or []), \
                "Dataset has %d train targets, while model expects %d" % \
                (len(raise_to_list(dataset.train_targets) or []), n_model_targets)
        # valid
        if dataset.valid_inputs is not None:
            assert n_model_inputs == len(raise_to_list(dataset.valid_inputs)), \
                "Dataset has %d valid inputs, while model expects %d" % \
                (len(raise_to_list(dataset.valid_inputs)), n_model_inputs)
            if not self.unsupervised:
                assert n_model_targets == len(raise_to_list(dataset.valid_targets) or []), \
                    "Dataset has %d valid targets, while model expects %d" % \
                    (len(raise_to_list(dataset.valid_targets) or []), n_model_targets)
        # test
        if dataset.test_inputs is not None:
            assert n_model_inputs == len(raise_to_list(dataset.test_inputs)), \
                "Dataset has %d test inputs, while model expects %d" % \
                (len(raise_to_list(dataset.test_inputs)), n_model_inputs)
            if not self.unsupervised:
                assert n_model_targets == len(raise_to_list(dataset.test_targets) or []), \
                    "Dataset has %d test targets, while model expects %d" % \
                    (len(raise_to_list(dataset.test_targets) or []), n_model_targets)

        # now we are happy, we can add them to `self`
        self.model = model
        self.dataset = dataset

        # Learning rate - how drastic of a step do the parameters change
        self.learning_rate = sharedX(learning_rate, 'learning_rate')
        # whether to scale individual model parameters' learning rates.
        self.lr_scalers = self.model.get_lr_scalers()
        # whether to decay
        if lr_decay:
            self.learning_rate_decay = get_decay_function(lr_decay,
                                                          self.learning_rate,
                                                          learning_rate,
                                                          lr_decay_factor)
        else:
            self.learning_rate_decay = False

        # rest of initial parameters needed for training.
        self.noise_switches = raise_to_list(self.model.get_noise_switch())
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.n_epoch = epochs
        self.save_frequency = save_freq
        self.early_stop_threshold = stop_threshold
        self.early_stop_length = stop_patience
        self.grad_clip = grad_clip
        self.hard_clip = hard_clip

    def get_updates(self, gradients):
        """
        This returns the parameter updates to use during training. It defaults to only using (annealed) learning rate.

        Parameters
        ----------
        gradients : dict
            A dictionary mapping from the model's parameters to their gradients.

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

    def train(self, monitor_channels=None, train_outservice=None, plot=None, additional_cost=None):
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
        additional_cost : theano expression or list(theano expression), optional
            Any additional cost expressions to use during training (things like regularization). These will be summed
            with the existing cost.
        """
        if not self.model:
            log.error("No self.model for the Optimizer!")
            raise AssertionError("Needs to be initialized with a Model! (Or something went wrong if train() "
                                 "was called from the Model. Try initializing the Optimizer with the model param "
                                 "and calling optimizer.train().")

        #####################################################
        # handle additional costs (normally regularization) #
        #####################################################
        # Create the gradient updates for the model - make sure to handle the possible
        # list of costs used for pretraining of certain parts of the model.
        train_costs = raise_to_list(self.model.get_train_cost())
        # deal with any other additional costs (like regularization, etc.)
        if additional_cost is not None:
            additional_costs = raise_to_list(additional_cost)
            if len(additional_costs) > 1:
                additional_cost = T.sum(additional_costs)

        #########################
        # gradients and updates #
        #########################
        train_updates = []
        self.gradients = []
        for i, train_cost in enumerate(train_costs):
            # Now create the training cost function for the model to use while training - update parameters
            # gradient!
            if len(train_costs) > 1 and additional_cost is not None:
                log.warning("additional_cost will double count with gradients during layer-wise pretraining!")
                warnings.warn("additional_cost will double count with gradients during layer-wise pretraining!")
            # TODO: additional_cost will double count with gradients during layer-wise pretraining.
            # Need to somehow make w.r.t. params appropriate for the individual training costs.
            gradients, _ = self.model.get_gradient(cost=train_cost, additional_cost=additional_cost)
            # clip gradients if we want.
            gradients = clip_gradients(gradients, self.grad_clip, self.hard_clip)
            # append to list
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

        ############
        # monitors #
        ############
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
        function_input = raise_to_list(self.model.get_inputs()) + raise_to_list(self.model.get_targets())
        train_functions = []
        for i, (updates, train_cost) in enumerate(zip(train_updates, train_costs)):
            # Compile the training function!
            log.info('Compiling f_learn %d/%d function for model %s...', i + 1, len(train_updates),
                     str(type(self.model)))
            t = time.time()

            f_learn = function(inputs=function_input,
                               updates=updates,
                               outputs=[train_cost] + list(self.train_monitors_dict.values()),
                               name='f_learn_%d' % i)

            log.info('f_learn %d compilation took %s', i + 1, make_time_units_string(time.time() - t))
            train_functions.append(f_learn)

        # figure out if we want valid and test (monitors)
        self.valid_flag = (self.dataset.valid_inputs is not None) and (len(self.valid_monitors_dict) > 0)
        self.test_flag = (self.dataset.test_inputs is not None) and (len(self.test_monitors_dict) > 0)
        # Now compile the monitor functions!
        log.debug("Compiling monitor functions...")
        monitor_t = time.time()
        # valid monitors
        if self.valid_flag:
            self.valid_monitor_function = function(
                inputs=function_input,
                updates=self.model.get_updates(),
                outputs=list(self.valid_monitors_dict.values()),
                name='valid_monitor_function'
            )
        else:
            self.valid_monitor_function = None

        # test monitors
        if self.test_flag:
            self.test_monitor_function = function(
                inputs=function_input,
                updates=self.model.get_updates(),
                outputs=list(self.test_monitors_dict.values()),
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
            log.info("-----------TRAINING %s function %d/%d FOR %d EPOCHS-----------",
                     str(type(self.model)), func_i + 1, len(train_functions), self.n_epoch)

            self.STOP = False
            self.epoch_counter = 0
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

        #########
        # train #
        #########
        train_costs = []
        train_monitors = {key: [] for key in self.train_monitors_dict.keys()}
        train_data = [
            minibatch(input, self.batch_size, self.min_batch_size)
            for input in raise_to_list(self.dataset.train_inputs)
            ]
        if self.dataset.train_targets is not None and not self.unsupervised:
            train_data += [
                minibatch(target, self.batch_size, self.min_batch_size)
                for target in raise_to_list(self.dataset.train_targets)
                ]

        for batch in min_normalized_izip(*train_data):
            _outs = raise_to_list(f_learn(*batch))
            train_costs.append(_outs[0])
            # handle any user defined monitors
            if len(train_monitors) > 0:
                current_monitors = zip(self.train_monitors_dict.keys(), _outs[1:])
                for name, val in current_monitors:
                    val = numpy.asarray(val)
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
            self.train_outservice.write(mean_train, "train")
        for name, service in self.train_monitors_outservice_dict.items():
            if name in current_mean_monitors and service:
                service.write(current_mean_monitors[name], "train")
        # if there is a plot, also send them over!
        if plot:
            current_mean_monitors.update({TRAIN_COST_KEY: mean_train})
            plot.update_plots(epoch=self.epoch_counter, monitors=current_mean_monitors)

        # set the noise switches off for valid and test sets! we assume unseen data is noisy anyway :)
        if len(self.noise_switches) > 0 and (self.valid_flag or self.test_flag):
            log.debug("Turning off %s noise switches", str(len(self.noise_switches)))
            [switch.set_value(0.) for switch in self.noise_switches]

        #########
        # valid #
        #########
        self._compute_over_subset("valid", self.dataset.valid_inputs, self.dataset.valid_targets,
                                  self.valid_monitors_dict, self.valid_monitor_function,
                                  self.valid_monitors_outservice_dict, plot)

        ########
        # test #
        ########
        self._compute_over_subset("test", self.dataset.test_inputs, self.dataset.test_targets,
                                  self.test_monitors_dict, self.test_monitor_function,
                                  self.test_monitors_outservice_dict, plot)

        ###########
        # cleanup #
        ###########
        # check for early stopping on train costs
        cost = numpy.sum(train_costs)
        # if the cost improved, reset the patience and record the best cost.
        if cost < self.best_cost * self.early_stop_threshold:
            self.patience = 0
            self.best_cost = cost
            # save the parameters that made it the best
            self.best_params = get_shared_values(self.params)
        elif not numpy.isnan(cost):
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

    def _compute_over_subset(self, subset, inputs, targets,
                             monitors_dict, monitor_function, monitors_outservice_dict,
                             plot):
        inputs = raise_to_list(inputs)
        targets = raise_to_list(targets)
        if inputs is not None and len(monitors_dict) > 0:
            monitors = {key: [] for key in monitors_dict.keys()}
            data = [minibatch(input, self.batch_size, self.min_batch_size) for input in inputs]
            if targets is not None and not self.unsupervised:
                data += [minibatch(target, self.batch_size, self.min_batch_size) for target in targets]

            for batch in min_normalized_izip(*data):
                _outs = raise_to_list(monitor_function(*batch))
                current_monitors = zip(monitors_dict.keys(), _outs)
                for name, val in current_monitors:
                    val = numpy.asarray(val)
                    monitors[name].append(val)

            # get the mean values for the batches
            current_mean_monitors = {key: numpy.mean(vals, 0) for key, vals in monitors.items()}
            # log the mean values!
            log.info('%s monitors: %s', subset, str(current_mean_monitors))
            # send the values to their outservices
            for name, service in monitors_outservice_dict.items():
                if name in current_mean_monitors and service:
                    service.write(current_mean_monitors[name], "test")
            # if there is a plot, also send them over!
            if plot:
                plot.update_plots(epoch=self.epoch_counter, monitors=current_mean_monitors)

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


def clip_gradients(gradients, grad_clip=5., hard_clip=False):
    """
    This returns the gradient parameters clipped according to the grad_clip value given in initialization.

    As described here: http://www.reddit.com/r/MachineLearning/comments/31b6x8/gradient_clipping_rnns/

    Code mostly taken from https://github.com/kastnerkyle/minet/blob/master/minet/net.py

    Based on:

    Pascanu, Razvan, Tomas Mikolov, and Yoshua Bengio. "On the difficulty of training
            recurrent neural networks." arXiv preprint arXiv:1211.5063 (2012).

    Parameters
    ----------
    gradients : dict
        A dictionary mapping from the model's parameters to their
        gradients.
    grad_clip : float, optional
        How much to clip gradients (if at all).
    hard_clip : bool
        Whether to use hard clipping (keeping gradients at grad_clip level), or soft clipping (rescaling based
        on grad_clip).

    Returns
    -------
    clipgrads : dict
        A dictionary mapping from the model's parameters to their correctly clipped
        gradients. (If no self.grad_clip, this just returns the original `gradients` input parameter).
    """
    if grad_clip:
        gradients = gradients.items()
        params = [item[0] for item in gradients]
        grads = [item[1] for item in gradients]

        # Gradient clipping
        grad_norm = T.sqrt(sum([T.sqr(grad).sum() for grad in grads]))
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        grad_norm = T.sqrt(grad_norm)
        scaling_num = grad_clip
        scaling_den = T.maximum(grad_clip, grad_norm)

        if hard_clip:
            # do the NaN/inf trick
            grads = [T.switch(not_finite,
                              0.1 * param,
                              grad)
                     for param, grad in gradients]
            # hard clip gradients above or below grad_clip to be = grad_clip
            grads = [T.switch(T.ge(grad_norm, grad_clip),
                              T.sgn(grad) * grad_clip,
                              grad)
                     for grad in grads]
        else:
            # NaN/inf trick combined with scaling.
            grads = [T.switch(not_finite,
                              0.1 * param,
                              grad * (scaling_num / scaling_den))
                     for param, grad in gradients]

        clipgrads = OrderedDict(zip(params, grads))
        return clipgrads
    else:
        return gradients