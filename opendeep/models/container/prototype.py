"""
This module defines a container for quickly assembling multiple layers/models
together without needing to define a new :class:`Model` class. This should mainly be used
for experimentation, and then later you should make your creation into a new :class:`Model` class.
"""
# standard libraries
import logging
# third party libraries
from theano.compat.python2x import OrderedDict
from theano.tensor import TensorVariable
# internal references
from opendeep.models.model import Model
from opendeep.utils.misc import raise_to_list

log = logging.getLogger(__name__)


class Prototype(Model):
    """
    The :class:`Prototype` lets you add :class:`Model`s in sequence, where the first model takes your input
    and the last model gives your output.

    The :class:`Prototype` is an iterable class, so you can index specific models inside or iterate over them
    with a for loop.

    You can use an :class:`Optimizer` with the container as you would a :class:`Model` - makes training easy :)

    Attributes
    ----------
    models : list
        The list of :class:`Model` objects that make up the :class:`Prototype`.
    """
    def __init__(self, layers=None, config=None, outdir='outputs/prototype/'):
        """
        During initialization, use the optional config provided to pre-set up the models. This is used
        for repeatable experiments.

        .. todo:: Add the ability to create models list from the input config. Right now, it does nothing.

        Parameters
        ----------
        layers : list(:class:`Model`), optional
            A model or list of models to initialize the :class:`Prototype` with.
        config : dict or JSON/YAML filename, optional
            A configuration defining the multiple models/configurations for this container to have.
        outdir : str, optional
            The location to produce outputs from training or running the :class:`Prototype`.
        """
        # initialize superclass (model) with the config
        super(Prototype, self).__init__(config=config, outdir=outdir, layers=layers)

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
        This adds a :class:`Model` (or list of models) to the sequence that the :class:`Prototype` holds.

        By default, we want single models added sequentially to use the outputs of the previous model as its
        `inputs` (if no `inputs` was defined by the user).

        Examples
        --------
        Here is the sequential creation of an MLP (no `inputs_hook` have to be defined, `add()` takes care
        of it automatically::

            from opendeep.models.container import Prototype
            from opendeep.models.single_layer.basic import Dense, SoftmaxLayer
            mlp = Prototype()
            mlp.add(Dense(input_size=28*28, output_size=1000, activation='relu', noise='dropout', noise_level=0.5))
            mlp.add(Dense(output_size=512, activation='relu', noise='dropout', noise_level=0.5))
            mlp.add(SoftmaxLayer(output_size=10))

        Parameters
        ----------
        model : :class:`Model` or list(:class:`Model`)
            The model (or list of models) to add to the Prototype. In the case of a single model with no `inputs`,
            the Prototype will configure the `inputs` to take the previous model's output from `get_outputs()`.
        """
        # check if model is a single model (not a list of models)
        if isinstance(model, Model):
            # if there is a previous layer added (more than one model in the Prototype)
            if len(self.models) > 0:
                # check if inputs (and hiddens) wasn't already defined by the user - basically a blank slate
                if model.inputs is None and model.hiddens is None:
                    log.info('Overriding model %s with new inputs from previous layer!', model._classname)
                    # get the previous layer output size and expression
                    previous_out_sizes = raise_to_list(self.models[-1].output_size)
                    previous_outs      = raise_to_list(self.models[-1].get_outputs())
                    # create the inputs from the previous outputs
                    current_inputs = zip(previous_out_sizes, previous_outs)
                    # grab the current model class
                    model_class = type(model)
                    # make the model a new instance of the current model (same arguments) except new inputs_hook
                    model_args = model.args.copy()
                    model_args['inputs'] = current_inputs
                    new_model = model_class(**model_args)
                    # clean up allocated variables from old model
                    for param in model.get_params().values():
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
        This is called by the :class:`Optimizer` when creating the theano train function on the cost expressions.
        Therefore, these are the training function inputs! (Which is different
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
            model_inputs = raise_to_list(model.get_inputs())
            # go through each and find the ones that are tensors in their basic input form (i.e. don't have an owner)
            for input in model_inputs:
                # if it is a tensor
                if isinstance(input, TensorVariable):
                    # if it doesn't have an owner
                    if hasattr(input, 'owner') and input.owner is None:
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
            return raise_to_list(self.models[-1].get_outputs())
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

    def get_switches(self):
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
            noise_switches.extend(raise_to_list(model.get_switches()))
        return noise_switches

    def get_params(self):
        """
        This returns the list of theano shared variables that will be trained by the :class:`Optimizer`.
        These parameters are used in the gradient.

        This includes all of the parameters in every model in the Prototype, without duplication.

        Returns
        -------
        dict(str: SharedVariable)
            Dictionary of {string_name: theano shared variables} to be trained with an :class:`Optimizer`.
            These are the parameters to be trained.
        """
        params = OrderedDict()
        for model_index, model in enumerate(self.models):
            model_params = model.get_params()
            # append the parameters only if they aren't already in the list!
            for name, param in model_params.items():
                if param not in list(params.values()):
                    name = model._classname + '_%d_' % model_index + name
                    params[name] = param
        return params
