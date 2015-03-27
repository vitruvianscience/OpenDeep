"""
.. module:: container

This module defines a container for quickly assembling multiple layers/models
together without needing to define a new Model class. This should mainly be used
for experimentation, and then later you should make it into a new Model class.
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
# third party libraries
import theano.tensor as T
# internal references
from opendeep import function
from opendeep.models.model import Model
from opendeep.utils.misc import make_time_units_string, raise_to_list

log = logging.getLogger(__name__)

class Prototype(Model):
    """
    The Prototype lets you add Models in sequence, where the first model takes your input
    and the last model gives your output.

    You can use an Optimizer with the container as you would a Model - makes training easy :)
    """
    def __init__(self, config=None, layers=None):
        """
        During initialization, use the optional config provided to pre-set up the model. This is used
        for repeatable experiments.

        :param config: a configuration defining the multiple models/configurations for this container to have.
        :type config: a dictionary-like object or filename to JSON/YAML file.
        """
        # initialize superclass (model) with the config
        super(Prototype, self).__init__(config=config)

        # TODO: add ability to create the models list from the input config.

        if layers is None:
            # create an empty list of the models this container holds.
            self.models = []
        else:
            # otherwise, use the layers input during initialization (make sure to raise to list)
            layers = raise_to_list(layers)
            self.models = layers

    def __getitem__(self, item):
        # let someone access a specific model in this container with indexing.
        return self.models[item]

    def __iter__(self):
        # let someone iterate through this container's models.
        for model in self.models:
            yield model

    def add(self, model):
        """
        This adds a model to the sequence that the container holds.

        By default, we want single models added sequentially to use the outputs of the previous model as its
        inputs_hook (if no inputs_hook was defined by the user)

        :param model: the model (or list of models) to add
        :type model: opendeep.models.Model
        """
        # check if model is a single model (not a list of models)
        if isinstance(model, Model):
            # if there is a previous layer added (more than one model in the Prototype)
            if len(self.models) > 0:
                # check if inputs_hook (and hiddens_hook) wasn't already defined by the user - basically a blank slate
                if model.inputs_hook is None and model.hiddens_hook is None:
                    log.info('Overriding model %s with new inputs_hook!', str(type(model)))
                    # get the previous layer output size and expression
                    previous_out_size   = self.models[-1].output_size
                    previous_out        = self.models[-1].get_outputs()
                    # create the inputs_hook from the previous outputs
                    current_inputs_hook = (previous_out_size, previous_out)
                    # grab the current model class
                    model_class = type(model)
                    # make the model a new instance of the current model (same arguments) except new inputs_hook
                    model_args = model.args
                    model_args['inputs_hook'] = current_inputs_hook
                    # no need for hiddens_hook - it must be None to be at this point.
                    model_args.pop('hiddens_hook')
                    new_model = model_class(**model_args)
                    # clean up allocated variables from old model
                    for param in model.get_params():
                        del param
                    del model
                    model = new_model

        # we want to be able to add multiple layers at a time (in a list), so using extend.
        # make sure the model is a list
        model = raise_to_list(model)
        self.models.extend(model)

    def get_inputs(self):
        """
        This should return the input(s) to the container's computation graph as a list.
        This is called by the Optimizer when creating the theano train function on the cost expressions
        returned by get_train_cost(). Therefore, these are the training function inputs! (Which is different
        from f_predict inputs if you include the supervised labels)

        This should normally return the same theano variable list that is used in the inputs= argument to the f_predict
        function for unsupervised models, and the [inputs, label] variables for the supervised case.
        ------------------

        :return: Theano variables representing the input(s) to the training function.
        :rtype: List(symbolic tensor)
        """
        inputs = []
        for model in self.models:
            # grab the inputs list from the model
            model_inputs = model.get_inputs()
            # go through each and find the ones that are tensors in their basic input form (i.e. don't have an owner)
            for input in model_inputs:
                # if it is a tensor
                if isinstance(input, T.TensorVariable) and hasattr(input, 'owner'):
                    # if it doesn't have an owner
                    if input.owner is None:
                        # add it to the running inputs list
                        input = raise_to_list(input)
                        inputs.extend(input)
        return inputs

    def get_outputs(self):
        """
        This method will return the container's output variable expression from the computational graph.
        This should be what is given for the outputs= part of the 'f_predict' function from self.predict().

        This will be used for creating hooks to link models together,
        where these outputs can be strung as the inputs or hiddens to another model :)

        Example: gsn = GSN()
                 softmax = SoftmaxLayer(inputs_hook=gsn.get_outputs())
        ------------------

        :return: theano expression of the outputs from this model's computation
        :rtype: theano tensor (expression)
        """
        # if this container has models, return the outputs to the very last model.
        if len(self.models) > 0:
            return self.models[-1].get_outputs()
        # otherwise, warn the user and return None
        else:
            log.warning("This container doesn't have any models! So no outputs to get...")
            return None

    def predict(self, input):
        """
        This method will return the model's output (run through the function), given an input. In the case that
        input_hooks or hidden_hooks are used, the function should use them appropriately and assume they are the input.

        Try to avoid re-compiling the theano function created for predict - check a hasattr(self, 'f_predict') or
        something similar first. I recommend creating your theano f_predict in a create_computation_graph method
        to be called after the class initializes.
        ------------------

        :param input: Theano/numpy tensor-like object that is the input into the model's computation graph.
        :type input: tensor

        :return: Theano/numpy tensor-like object that is the output of the model's computation graph.
        :rtype: tensor
        """
        # make sure the input is raised to a list - we are going to splat it!
        input = raise_to_list(input)
        # first check if we already made an f_predict function
        if hasattr(self, 'f_predict'):
            return self.f_predict(*input)
        # otherwise, compile it!
        else:
            inputs = self.get_inputs()
            outputs = self.get_outputs()
            updates = self.get_updates()
            t = time.time()
            log.info("Compiling f_predict...")
            self.f_predict = function(inputs=inputs, outputs=outputs, updates=updates, name="f_predict")
            log.info("Compilation done! Took %s", make_time_units_string(time.time() - t))
            return self.f_predict(*input)

    def get_targets(self):
        """
        This grabs the targets (for supervised training) of the last layer in the model list.

        :return: list(symbolic tensor)
        """
        # if this container has models, return the targets to the very last model.
        if len(self.models) > 0:
            return self.models[-1].get_targets()
        # otherwise, warn the user and return None
        else:
            log.warning("This container doesn't have any models! So no targets to get...")
            return None

    def get_train_cost(self):
        """
        This returns the expression that represents the cost given an input, which is used for the Optimizer during
        training. The reason we can't just compile a f_train theano function is because updates need to be calculated
        for the parameters during gradient descent - and these updates are created in the Optimizer object.

        In the specialized case of layer-wise pretraining (or any version of pretraining in the model), you should
        return a list of training cost expressions in order you want training to happen. This way the optimizer
        will train each cost in sequence for your model, allowing for easy layer-wise pretraining in the model.
        ------------------

        :return: theano expression (or list of theano expressions)
        of the model's training cost, from which parameter gradients will be computed.
        :rtype: theano tensor or list(theano tensor)
        """
        # if this container has models, return the outputs to the very last model.
        if len(self.models) > 0:
            return self.models[-1].get_train_cost()
        # otherwise, warn the user and return None
        else:
            log.warning("This container doesn't have any models! So no outputs to get...")
            return None

    def get_gradient(self, starting_gradient=None, cost=None, additional_cost=None):
        """
        This method allows you to define the gradient for this model manually. It should either work with a provided
        starting gradient (from upstream layers/models), or grab the training cost if no start gradient is provided.

        Theano's subgraph gradient function specified here:
        http://deeplearning.net/software/theano/library/gradient.html#theano.gradient.subgraph_grad
        warning: If the gradients of cost with respect to any of the start variables is already part of the
        start dictionary, then it may be counted twice with respect to wrt and end.

        You should only implement this method if you want to manually define your gradients for the model.
        --------------------

        :param starting_gradient: the starting, known gradients for variables
        :type starting_gradient: dictionary of {variable: known_gradient}

        :param additional_cost: any additional cost to add to the gradient
        :type additional_cost: theano expression

        :return: tuple of gradient with respect to inputs, and with respect to
        :rtype:
        """
        # for now just use the Model's get_gradient method.
        return super(Prototype, self).get_gradient(starting_gradient, cost, additional_cost)

    def get_updates(self):
        """
        This should return any theano updates from the models (used for things like random number generators).
        Most often comes from theano's 'scan' op. Check out its documentation at
        http://deeplearning.net/software/theano/library/scan.html.

        This is used with the optimizer to create the training function - the 'updates=' part of the theano function.
        ------------------

        :return: updates from the theano computation for the model to be used during Optimizer.train()
        (but not including training parameter updates - those are calculated by the Optimizer)
        These are expressions for new SharedVariable values.
        :rtype: (iterable over pairs (shared_variable, new_expression). List, tuple, or dict.)
        """
        # Return the updates going through each model in the list:
        updates = None
        for model in self.models:
            current_updates = model.get_updates()
            # if updates exist already and the current model in the list has updates, update accordingly!
            if updates and current_updates:
                updates.update(current_updates)
            # otherwise if there haven't been updates yet but the current model has them, set as the base updates.
            elif current_updates:
                updates = current_updates
        return updates

    def get_monitors(self):
        """
        This returns a dictionary of (monitor_name: monitor_function) of variables (monitors) whose values we care
        about during training. For every monitor returned by this method, the function will be run on the
        train/validation/test dataset and its value will be reported.

        Again, please avoid recompiling the monitor functions every time - check your hasattr to see if they already
        exist!
        ------------------

        :return: Dictionary of String: theano_function for each monitor variable we care about in the model.
        :rtype: Dictionary
        """
        # Return the monitors going through each model in the list:
        monitors = {}
        for model in self.models:
            current_monitors = model.get_monitors()
            monitors.update(current_monitors)
        return monitors

    def get_decay_params(self):
        """
        If the model requires any of its internal parameters to decay over time during training, return the list
        of the DecayFunction objects here so the Optimizer can decay them each epoch. An example is the noise
        amount in a Generative Stochastic Network - we decay the noise over time when implementing noise scheduling.

        Most models don't need to decay parameters, so we return an empty list by default. Please override this method
        if you need to decay some variables.
        ------------------

        :return: List of opendeep.utils.decay_functions.DecayFunction objects of the parameters to decay for this model.
        :rtype: List
        """
        # Return the decay params going through each model in the list:
        decay_params = []
        for model in self.models:
            decay_params.extend(model.get_decay_params())
        return decay_params

    def get_lr_scalers(self):
        """
        This method lets you scale the overall learning rate in the Optimizer to individual parameters.
        Returns a dictionary mapping model_parameter: learning_rate_scaling_factor. Default is no scaling.
        ------------------

        :return: dictionary mapping the model parameters to their learning rate scaling factor
        :rtype: Dictionary(shared_variable: float)
        """
        # Return the lr scalers going through each model in the list
        lr_scalers = {}
        for model in self.models:
            lr_scalers.update(model.get_lr_scalers())
        return lr_scalers

    def get_params(self):
        """
        This returns the list of theano shared variables that will be trained by the Optimizer.
        These parameters are used in the gradient.
        ------------------

        :return: flattened list of theano shared variables to be trained
        :rtype: List(shared_variables)
        """
        # Return the decay params going through each model in the list:
        params = []
        for model in self.models:
            model_params = model.get_params()
            # append the parameters only if they aren't already in the list!
            # using a set would lose the order, which is important.
            for param in model_params:
                if param not in params:
                    params.append(param)
        return params