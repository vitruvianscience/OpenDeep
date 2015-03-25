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
from opendeep.utils.misc import make_time_units_string

log = logging.getLogger(__name__)

class Prototype(Model):
    """
    The Prototype lets you add Models in sequence, where the first model takes your input
    and the last model gives your output.

    You can use an Optimizer with the container as you would a Model - makes training easy :)
    """
    def __init__(self, config=None):
        """
        During initialization, use the optional config provided to pre-set up the model. This is used
        for repeatable experiments.

        :param config: a configuration defining the multiple models/configurations for this container to have.
        :type config: a dictionary-like object or filename to JSON/YAML file.
        """
        # create an empty list of the models this container holds.
        self.models = []
        # TODO: add ability to create the models list from the input config.

    def __getitem__(self, item):
        # let someone access a specific model in this container
        return self.models[item]

    def __iter__(self):
        # let someone iterate through this container's models
        for model in self.models:
            yield model

    def add(self, model):
        """
        This adds a model to the sequence that the container holds.

        :param model: the model to add
        :type model: opendeep.models.Model
        """
        self.models.append(model)

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
        :rtype: List(theano variable)
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
        # first check if we already made an f_predict function
        if hasattr(self, 'f_predict'):
            return self.f_predict(input)
        # otherwise, compile it!
        else:
            inputs  = self.get_inputs()
            outputs = self.get_outputs()
            updates = self.get_updates()
            t = time.time()
            log.info("Compiling f_predict...")
            self.f_predict = function(inputs=inputs, outputs=outputs, updates=updates, name="f_predict")
            log.info("Compilation done! Took ", make_time_units_string(time.time() - t))
            return self.f_predict(input)

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