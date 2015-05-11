"""
This module defines a container for quickly assembling multiple layers/models
together without needing to define a new Model class. This should mainly be used
for experimentation, and then later you should make your creation into a new Model class.
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

    You can use an :class:`Optimizer` with the container as you would a Model - makes training easy :)

    Attributes
    ----------
    models : list
        The list of :class:`Model` objects that make up the :class:`Prototype`.
    """
    def __init__(self, config=None, layers=None, outdir='outputs/prototype/'):
        """
        During initialization, use the optional config provided to pre-set up the models. This is used
        for repeatable experiments.

        .. todo:: Add the ability to create models list from the input config. Right now, it does nothing.

        Parameters
        ----------
        config : dict or JSON/YAML filename, optional
            A configuration defining the multiple models/configurations for this container to have.
        layers : list(:class:`Model`)
            A model or list of models to initialize the :class:`Prototype` with.
        outdir : str
            The location to produce outputs from training or running the :class:`Prototype`.
        """
        # initialize superclass (model) with the config
        super(Prototype, self).__init__(config=config, outdir=outdir)

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
        This adds a model (or list of models) to the sequence that the :class:`Prototype` holds.

        By default, we want single models added sequentially to use the outputs of the previous model as its
        `inputs_hook` (if no `inputs_hook` was defined by the user).

        Examples
        --------
        Here is the sequential creation of an MLP (no `inputs_hook` have to be defined, `add()` takes care
        of it automatically::

            from opendeep.models.container import Prototype
            from opendeep.models.single_layer.basic import BasicLayer, SoftmaxLayer
            mlp = Prototype()
            mlp.add(BasicLayer(input_size=28*28, output_size=1000, activation='relu', noise='dropout', noise_level=0.5))
            mlp.add(BasicLayer(output_size=512, activation='relu', noise='dropout', noise_level=0.5))
            mlp.add(SoftmaxLayer(output_size=10))

        Parameters
        ----------
        model : :class:`Model` or list(:class:`Model`)
            The model (or list of models) to add to the Prototype. In the case of a single model with no `inputs_hook`,
            the Prototype will configure the `inputs_hook` to take the previous model's output from `get_outputs()`.
        """
        # check if model is a single model (not a list of models)
        if isinstance(model, Model):
            # if there is a previous layer added (more than one model in the Prototype)
            if len(self.models) > 0:
                # check if inputs_hook (and hiddens_hook) wasn't already defined by the user - basically a blank slate
                if model.inputs_hook is None and model.hiddens_hook is None:
                    log.info('Overriding model %s with new inputs_hook!', str(type(model)))
                    # get the previous layer output size and expression
                    previous_out_size = self.models[-1].output_size
                    previous_out      = self.models[-1].get_outputs()
                    # create the inputs_hook from the previous outputs
                    current_inputs_hook = (previous_out_size, previous_out)
                    # grab the current model class
                    model_class = type(model)
                    # make the model a new instance of the current model (same arguments) except new inputs_hook
                    model_args = model.args.copy()
                    model_args['inputs_hook'] = current_inputs_hook
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
        This should return the input(s) to the Prototype's computation graph as a list.
        This is called by the :class:`Optimizer` when creating the theano train function on the cost expressions
        returned by get_train_cost(). Therefore, these are the training function inputs! (Which is different
        from f_run inputs if you include the supervised labels)

        This gets a list of all unique inputs by going through each model in the Prototype and checking if its
        inputs are used as hooks to other models or are unique (a starting point in the computation graph).

        Returns
        -------
        List(tensor)
            Theano variables representing the input(s) to the computation graph.
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
        This should be what is given for the `outputs=` part of the 'f_run' function from `self.run()`.

        This comes from the last model in Prototype's `get_outputs()` function.

        Returns
        -------
        tensor
            Theano expression of the outputs from this Prototype's computation
        """
        # if this container has models, return the outputs to the very last model.
        if len(self.models) > 0:
            return self.models[-1].get_outputs()
        # otherwise, warn the user and return None
        else:
            log.warning("This container doesn't have any models! So no outputs to get...")
            return None

    def run(self, input):
        """
        This method will return the Prototype's output (run through the `f_run` function), given an input. The input
        comes from all unique inputs to the models in the Prototype as calculated from `get_inputs()` and the outputs
        computed similarly from `get_outputs`.

        Try to avoid re-compiling the theano function created for run - check a `hasattr(self, 'f_run')` or
        something similar first.

        Parameters
        ----------
        input: array_like
            Theano/numpy tensor-like object that is the input into the model's computation graph.

        Returns
        -------
        array_like
            Theano/numpy tensor-like object that is the output of the model's computation graph.
        """
        # make sure the input is raised to a list - we are going to splat it!
        input = raise_to_list(input)
        # first check if we already made an f_run function
        if hasattr(self, 'f_run'):
            return self.f_run(*input)
        # otherwise, compile it!
        else:
            inputs = self.get_inputs()
            outputs = self.get_outputs()
            updates = self.get_updates()
            t = time.time()
            log.info("Compiling f_run...")
            self.f_run = function(inputs=inputs, outputs=outputs, updates=updates, name="f_run")
            log.info("Compilation done! Took %s", make_time_units_string(time.time() - t))
            return self.f_run(*input)

    def get_targets(self):
        """
        This grabs the targets (for supervised training) of the last layer in the model list to use for training.

        Returns
        -------
        list(tensor)
            List of the target variables to use during training. Comes from the last model's `get_targets()` function.
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
        This returns the expression that represents the cost given an input, which is used for the :class:`Optimizer`
        during training. The reason we can't just compile a f_train theano function is because updates need to be
        calculated for the parameters during gradient descent - and these updates are created in the :class:`Optimizer`
        object.

        Similar to other methods in the Prototype, this returns the last model's cost from `train_cost()`.

        Returns
        -------
        tensor or list(tensor)
            Theano expression (or list of theano expressions) of the model's training cost,
            from which parameter gradients will be computed. Comes from calling `get_train_cost()` on
            the last model in the Prototype.
        """
        # if this container has models, return the outputs to the very last model.
        if len(self.models) > 0:
            return self.models[-1].get_train_cost()
        # otherwise, warn the user and return None
        else:
            log.warning("This container doesn't have any models! So no outputs to get...")
            return None

    def get_updates(self):
        """
        This should return any theano updates from the models (used for things like random number generators).
        Most often comes from theano's 'scan' op. Check out its documentation at
        http://deeplearning.net/software/theano/library/scan.html.

        This is used with the optimizer to create the training function - the `updates=` part of the theano function.

        The updates come from calling `get_updates()` on every model in the Prototype.

        Returns
        -------
        iterable over pairs (SharedVariable, new_expression).
            Updates from the theano computation for the model to be used during Optimizer.train()
            (but not including training parameter updates - those are calculated by the Optimizer)
            These are expressions for new SharedVariable values.
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
        This returns a dictionary of (monitor_name: monitor_expression) of variables (monitors) whose values we care
        about during training.

        .. note::
            This is created by updating a dictionary with the monitors returned by `get_monitors()` called on every
            model in the Prototype. This means that models that have monitors with the same name will override
            previous expression values.

        Returns
        -------
        dict
            Dictionary of String: theano_expression for each monitor variable we would care about in the Prototype.
        """
        # Return the monitors going through each model in the list:
        monitors = {}
        for model in self.models:
            current_monitors = model.get_monitors()
            monitors.update(current_monitors)
        return monitors

    def get_decay_params(self):
        """
        If the Prototype requires any of its internal parameters to decay over time during training, return the list
        of the DecayFunction objects here so the :class:`Optimizer` can decay them each epoch.

        Again, this is calculated by calling `get_decay_params()` on every model in the Prototype.

        Returns
        -------
        list
            List of opendeep.utils.decay_functions.DecayFunction objects of the parameters to decay for this Prototype.
        """
        # Return the decay params going through each model in the list:
        decay_params = []
        for model in self.models:
            decay_params.extend(model.get_decay_params())
        return decay_params

    def get_lr_scalers(self):
        """
        This method lets you scale the overall learning rate in the :class:`Optimizer` to individual parameters.
        Returns a dictionary mapping {model_parameter: learning_rate_scaling_factor}. Default is no scaling.

        .. note::
            This is created by updating a dictionary with `get_lr_scalers()` called on every
            model in the Prototype. This means that shared parameters could be overriden.

        Returns
        -------
        dict
            Dictionary of (SharedVariable: float) mapping the model parameters to their learning rate scaling factor.
        """
        # Return the lr scalers going through each model in the list
        lr_scalers = {}
        for model in self.models:
            lr_scalers.update(model.get_lr_scalers())
        return lr_scalers

    def get_noise_switch(self):
        """
        This method returns a list of shared theano variables representing switches for adding noise in the model.

        This is constructed by calling `get_noise_switch()` on every model in the Prototype.

        Returns
        -------
        list
            List of the shared variables representing switches to be turned on during training and off during f_run.
        """
        # Return the noise switches going through each model in the list
        noise_switches = []
        for model in self.models:
            noise_switches.extend(raise_to_list(model.get_noise_switch()))
        return noise_switches

    def get_params(self):
        """
        This returns the list of theano shared variables that will be trained by the :class:`Optimizer`.
        These parameters are used in the gradient.

        This includes all of the parameters in every model in the Prototype, without duplication.

        Returns
        -------
        list
            Flattened list of theano shared variables to be trained.
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