"""
This module defines the generic Model class -
which represents everything from a single layer to a full-blown deep network.

Models are the reusable, modular building blocks for deep networks. Their power comes from
their ability to connect with other Models.
"""

__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
import os
import time
# third party libraries
import theano
import theano.tensor as T
from theano.compat.python2x import OrderedDict  # use this compatibility OrderedDict
# internal references
from opendeep import function
from opendeep.utils import file_ops
from opendeep.utils.misc import set_shared_values, get_shared_values, make_time_units_string, raise_to_list, add_kwargs_to_dict
from opendeep.utils.file_ops import mkdir_p

try:
    import cPickle as pickle
except ImportError:
    import pickle

log = logging.getLogger(__name__)


class Model(object):
    """
    The :class:`Model` is a generic class for everything from a single layer to complex multi-layer behemoths
    (which can be a combination of multiple models linked through input_hooks, hidden_hooks, and params_hooks).

    Think of a :class:`Model` like Legos - you can attach single pieces together as well as multi-piece units together.
    The main vision of OpenDeep is to provide a lightweight, highly modular structure that makes creating and
    experimenting with new models as easy as possible. Much of current deep learning progress has come from
    combining multiple deep models together for complex tasks.

    When creating Theano functions inside of models, use the `opendeep.function` wrapper instead of the basic
    `theano.function` - this changes unused inputs from an error to a warning. Most likely, unused inputs
    shouldn't be a breaking error.

    Attributes
    ----------
    args : dict
        This is a dictionary containing all the input parameters that initialize the Model. Think of it
        as the configuration for initializing a :class:`Model`.
    inputs_hook : tuple
        Tuple of (shape, input_variable) or None describing the inputs to use for this Model.
    hiddens_hook : tuple
        Tuple of (shape, hiddens_variable) or None to use as the hidden representation for this Model.
    params_hook : list
        A list of SharedVariable representing the parameters to use for this Model.
    input_size : int or shape tuple
        The dimensionality of the input for this model. This is required for stacking models
        automatically - where the input to one layer is the output of the previous layer.
    output_size : int or shape tuple
        Describes the shape of the output dimensionality for this Model.
    outdir : str
        The filepath to save outputs for this Model (such as pickled parameters created during training,
        visualizations, etc.).
    f_run : function
        Theano function for running the model's computation on an input. This gets set during compile_run_fn().
    """

    def __init__(self, inputs_hook=None, hiddens_hook=None, params_hook=None,
                 input_size=None, output_size=None,
                 outdir=None,
                 **kwargs):
        """
        Initialize a new Model.

        Your model implementations should accept optional inputs_hook and hiddens_hook (if applicable)
        to set your inputs and hidden representation in a modular fashion, allowing models to link together.
        inputs_hook is a tuple of (shape, variable) that should replace the default model inputs.
        hiddens_hook is a tuple of (shape, variable) that should replace the default model hidden representation
        (which means you need to adapt creating your computation graph to not care about the inputs and to instead
        run outputs directly from the hidden variable provided).
        You can also accept a params_hook to share model parameters rather than instantiate a new set of parameters.

        Parameters
        ----------
        inputs_hook : Tuple of (shape, variable)
            Routing information for the model to accept inputs from elsewhere. This is used for linking
            different models together (e.g. setting the Softmax model's input layer to the DAE's hidden layer gives a
            newly supervised classification model). For now, it needs to include the shape information (normally the
            dimensionality of the input i.e. n_in).
        hiddens_hook : Tuple of (shape, variable)
            Routing information for the model to accept its hidden representation from elsewhere.
            This is used for linking different models together (e.g. setting the GSN model's hidden layers to the RNN's
            output layer gives the RNN-GSN model, a deep recurrent model.) For now, it needs to include the shape
            information (normally the dimensionality of the hiddens i.e. n_hidden).
        params_hook : List(theano shared variable)
            A list of model parameters (shared theano variables) that you should use when constructing
            this model (instead of initializing your own shared variables). This parameter is useful when you want to
            have two versions of the model that use the same parameters - such as a training model with dropout applied
            to layers and one without for testing, where the parameters are shared between the two.
        input_size : int or shape tuple
            The dimensionality of the input for this model. This is required for stacking models
            automatically - where the input to one layer is the output of the previous layer.
        output_size : int or shape tuple
            The dimensionality of the output for this model. This is required for stacking models
            automatically - where the input to one layer is the output of the previous layer. Currently, we cannot
            run the size from Theano's graph, so it needs to be explicit.
        outdir : str
            The directory you want outputs (parameters, images, etc.) to save to. If None, nothing will
            be saved.
        kwargs : dict
            This will be all the other left-over keyword parameters passed to the class as a
            dictionary of {param: value}. These get created into `self.args` along with outdir and output_size.
        """
        log.info("Creating a new instance of %s", str(type(self)))

        # Necessary inputs to a Model - these are the minimum requirements for modularity to work.
        self.inputs_hook  = inputs_hook
        self.hiddens_hook = hiddens_hook
        self.params_hook  = params_hook
        self.input_size   = input_size
        self.output_size  = output_size
        self.outdir       = outdir

        # make sure outdir ends in a directory separator
        if self.outdir and self.outdir[-1] != os.sep:
            self.outdir += os.sep

        # Combine arguments that could specify input_size -> overwrite input_size with inputs_hook[0] if it exists.
        if self.inputs_hook and self.inputs_hook[0] is not None:
            self.input_size = self.inputs_hook[0]

        # Check if the input_size wasn't provided - if this is the case, it could either be a programmer's error
        # or it could be during the automatic stacking in a Container. Since that is a common use case, set
        # the input_size to 1 to avoid errors when instantiating the model.
        if not self.input_size:
            # Could be error, or more commonly, when adding models to a Container
            log.warning("No input_size or inputs_hook! Make sure this is done in a Container. Setting input_size"
                        "=1 for the Container now...")
            self.input_size = 1

        # Also, check if no output_size was given - this could be the case for generative models. Copy input_size
        # in that case.
        if not self.output_size:
            # Could be an error (hopefully not), so give the warning.
            log.warning("No output_size given! Make sure this is from a generative model (where output_size is the"
                        "same as input_size. Setting output_size=input_size now...")
            self.output_size = self.input_size

        # copy all of the parameters from the class into an args (configuration) dictionary
        self.args = {}
        self.args = add_kwargs_to_dict(kwargs.copy(), self.args)

        self.args['output_size'] = self.output_size

        # Now create the directory for outputs of the model
        # set up base path for the outputs of the model during training, etc.
        self.args['outdir'] = self.outdir
        if self.args['outdir']:
            mkdir_p(self.args['outdir'])

        # log the arguments.
        log.info("%s self.args: %s", str(type(self)), str(self.args))
        # save the arguments.
        self.save_args()
        # Boom! Hyperparameters are now dealt with. Take that!

    ######################################################################
    # Methods for the symbolic inputs, hiddens, and outputs of the model #
    ######################################################################
    def get_inputs(self):
        """
        This should return the input(s) to the model's computation graph as a list. This only includes inputs for the
        run function, not any inputs used for supervised training.

        .. note::
            This should normally return the same theano variable list that is used in the inputs= argument to the
            f_run function when running the Model on an input.

        Returns
        -------
        Theano variable or List(theano variable)
            Theano variables representing the input(s) to the model's 'run' computation.

        Raises
        ------
        NotImplementedError
            If the function hasn't been implemented for the specific model.
        """
        log.critical("%s does not have a get_inputs function!", str(type(self)))
        raise NotImplementedError("Please implement a get_inputs method for %s" % str(type(self)))

    def get_hiddens(self):
        """
        This method will return the model's hidden representation expression (if applicable)
        from the computational graph. This is normally useful for unsupervised models, whose hidden units
        learn the representation of the input.

        This will also be used for creating hooks to link models together, where these hidden variables can be strung
        as the inputs or hiddens to another model :)

        Returns
        -------
        theano expression
            Theano expression of the hidden representation from this model's computation.

        Raises
        ------
        NotImplementedError
            If the function hasn't been implemented for the specific model.
        """
        log.critical("%s get_hiddens method not implemented!", str(type(self)))
        raise NotImplementedError("Please implement a get_hiddens method for %s" % str(type(self)))

    def get_outputs(self):
        """
        This method will return the model's output variable expression from the computational graph.
        This should be what is given for the outputs= part of the 'f_run' function from `self.run()`.

        This will be used for creating hooks to link models together,
        where these outputs can be strung as the inputs or hiddens to another model :)

        Returns
        -------
        theano expression
            Theano expression of the outputs from this model's computation graph.

        Examples
        --------
        Here is an example showing the `get_outputs()` method in the GSN model used in an `inputs_hook`
        to a SoftmaxLayer model::

            from opendeep.models.multi_layer.generative_stochastic_network import GSN
            from opendeep.models.single_layer.basic import SoftmaxLayer
            gsn = GSN(input_size=28*28, hidden_size=1000, layers=2, walkbacks=4)
            softmax = SoftmaxLayer(inputs_hook=(gsn.output_size, gsn.get_outputs()), output_size=10)

        Raises
        ------
        NotImplementedError
            If the function hasn't been implemented for the specific model.
        """
        log.critical("%s get_outputs method not implemented!", str(type(self)))
        raise NotImplementedError("Please implement a get_outputs method for %s" % str(type(self)))

    #############################################
    # Methods for running the model on an input #
    #############################################
    def compile_run_fn(self):
        """
        This is a helper function to compile the f_run function for computing the model's outputs given inputs.
        Compile and set the f_run function used for `run()`.

        It sets the `self.f_run` attribute to the f_run function.

        .. note::
            The run function defaults like so::

                self.f_run = function(inputs  = raise_to_list(self.get_inputs()),
                                      outputs = self.get_outputs(),
                                      updates = self.get_updates(),
                                      name    = 'f_run')
        """
        if not hasattr(self, 'f_run'):
            log.debug("Compiling f_run...")
            t = time.time()
            self.f_run = function(inputs  = raise_to_list(self.get_inputs()),
                                  outputs = self.get_outputs(),
                                  updates = self.get_updates(),
                                  name    = 'f_run')
            log.debug("Compilation done. Took %s", make_time_units_string(time.time() - t))
        else:
            log.warn('f_run already exists!')

    def run(self, input):
        """
        This method will return the model's output (run through the function), given an input. In the case that
        input_hooks or hidden_hooks are used, the function should use them appropriately and assume they are the input.

        .. note::
            If the Model doesn't have an f_run attribute,
            it will run `compile_run_fn()` to compile the appropriate function.

        Parameters
        ----------
        input : tensor
            Theano/numpy tensor-like object that is the input into the model's computation graph.

        Returns
        -------
        array_like
            Array_like object that is the output of the model's computation graph run on the given input.
        """
        # check if the run function is already compiled, otherwise compile it!
        if not hasattr(self, 'f_run'):
            self.compile_run_fn()

        # because we use the splat to account for multiple inputs to the function, make sure input is a list.
        input = raise_to_list(input)
        # return the results of the run function!
        output = self.f_run(*input)

        return output

    def generate(self, initial=None):
        """
        This method starts generating samples from the model (if it is a generative model).

        Parameters
        ----------
        initial : array_like, optional
            The starting point for generation (if applicable). Defaults to None.

        Returns
        -------
        list
            The list of generated values from the Model (starting from `initial` if applicable).

        Raises
        ------
        NotImplementedError
            If the function hasn't been implemented for the specific model.
        """
        log.exception("Generate method not implemented for Model %s", str(type(self)))
        raise NotImplementedError("Generate method not implemented for Model %s" % str(type(self)))

    #########################################
    # Methods to do with training the model #
    #########################################
    def get_targets(self):
        """
        This function returns the list of inputs that are used for supervised training. It should be the 'correct' or
        'target' variables to compare against the output of the model's computation. This method needs to be
        implemented for supervised models.

        Example: the labels Y for a classification problem.

        Returns
        -------
        theano variable or List(theano variable)
            Theano variables representing the target(s) to the model's computation. Defaults to returning an empty
            list, which assumes the model is unsupervised.

        """
        # Assume we have an unsupervised function, so no extra training variables. If this is going to be a supervised
        # model, you have to return the list of extra 'label' (aka 'target') variables you created for the cost
        # function here.
        return []

    def get_train_cost(self):
        """
        This returns the expression that represents the cost given an input, which is used during training.
        The reason we can't just compile an f_train theano function is because updates need to be calculated
        for the parameters during gradient descent - and these updates are created in the :class:`Optimizer` object.

        In the specialized case of layer-wise pretraining (or any version of pretraining in the model), you should
        return a list of training cost expressions in order you want training to happen. This way the optimizer
        will train each cost in sequence for your model, allowing for easy layer-wise pretraining in the model.

        Returns
        -------
        theano expression or list of theano expressions
            The model's training cost(s), from which parameter gradients will be computed.

        Raises
        ------
        NotImplementedError
            If the function hasn't been implemented for the specific model.
        """
        log.critical("%s does not have a get_train_cost function!", str(type(self)))
        raise NotImplementedError("Please implement a get_train_cost method for %s" % str(type(self)))

    def get_gradient(self, starting_gradient=None, cost=None, additional_cost=None):
        """
        This method allows you to define the gradient for this model manually. It should either work with a provided
        starting gradient (from upstream layers/models), or grab the training cost if no start gradient is provided.

        Theano's subgraph gradient function specified here:
        http://deeplearning.net/software/theano/library/gradient.html#theano.gradient.subgraph_grad

        .. warning::
            If the gradients of cost with respect to any of the start variables is already part of the
            start dictionary, then it may be counted twice with respect
            to wrt (`get_params()`) and end (`get_inputs()`).

        You should only implement this method if you want to manually define your gradients for the model.

        Parameters
        ----------
        starting_gradient : dictionary of {variable: known_gradient}, optional
            The starting, known gradients for parameters.
        cost : theano expression, optional
            The cost expression to use when calculating the gradients. Defaults to `get_train_cost()`.
        additional_cost : theano expression, optional
            Any additional cost to add to the gradient.

        Returns
        -------
        tuple
            (Gradient with respect to params, gradient with respect to inputs)
        """
        # check if starting gradients was provided.
        # if there are known gradients to start, use those instead of the cost for this model
        if starting_gradient is not None:
            params_grad, next_starting_grad = theano.subgraph_grad(wrt=self.get_params(),
                                                                   end=raise_to_list(self.get_inputs()),
                                                                   start=starting_gradient,
                                                                   cost=additional_cost,
                                                                   details=False)
        # otherwise, just use this model's cost to determine gradient
        else:
            # use the cost if it was given
            cost = cost or self.get_train_cost()
            if additional_cost is not None:
                cost = T.sum(cost, additional_cost)
            params_grad, next_starting_grad = theano.subgraph_grad(wrt=self.get_params(),
                                                                   end=raise_to_list(self.get_inputs()),
                                                                   cost=cost,
                                                                   details=False)
        return (OrderedDict(zip(self.get_params(), params_grad)),
                OrderedDict(zip(raise_to_list(self.get_inputs()), next_starting_grad)))

    def get_updates(self):
        """
        This should return any theano updates from the model (used for things like random number generators).
        Most often comes from theano's 'scan' op. Check out its documentation at
        http://deeplearning.net/software/theano/library/scan.html.

        This is used with the :class:`Optimizer` to create the training function - the 'updates='
        part of the theano function.

        Returns
        -------
        iterable over pairs (shared_variable, new_expression)
            Updates from the theano computation for the model to be used during Optimizer.train()
            (but not including training parameter updates - those are calculated by the :class:`Optimizer`)
            These are expressions for new SharedVariable values.
        """
        # TODO: should we do the parameter decays from get_decay_params() in the model updates?
        # TODO: Right now I'm not because it seems less modular
        # TODO: do we need a list of these as well to deal with the possible list of get_train_cost()?
        # by default, assume the model doesn't have updates - it's your job to return them in this method.
        return None

    def get_monitors(self):
        """
        This returns a dictionary of (monitor_name: monitor_expression) of variables (monitors) whose values we care
        about during training. Often times, this is a log-likelihood estimator, mean squared error, weights
        statistics, etc. The actual training cost value used in `get_train_cost()` does not need to be included -
        it will automatically be monitored.

        Returns
        -------
        dict
            Dictionary of {string name: theano expression} for each monitor variable we care about in the model.
        """
        # no monitors by default
        return {}

    def get_decay_params(self):
        """
        If the model requires any of its internal parameters to decay over time during training, return the list
        of the :class:`DecayFunction` (from opendeep.utils.decay) objects here so the :class:`Optimizer` can decay
        them each epoch. An example is the noise amount in a Generative Stochastic Network - we decay the noise
        over time when implementing noise scheduling.

        Most models don't need to decay parameters, so we return an empty list by default. Please override this method
        if you need to decay some variables.

        Returns
        -------
        list(:class:`DecayFunction`)
            List of opendeep.utils.decay_functions.DecayFunction objects of the parameters to decay for this model.
            Defaults to an empty list - no decay parameters.
        """
        # no decay parameters by default
        return []

    def get_lr_scalers(self):
        """
        This method lets you scale the overall learning rate in the :class:`Optimizer` to individual parameters.
        Returns a dictionary mapping model_parameter: learning_rate_scaling_factor. Default is an empty
        dictionary which means no scaling.

        Returns
        -------
        dict
            Dictionary mapping the model parameters to their learning rate scaling factor {SharedVariable: float}.
        """
        # By default, no learning rate scaling.
        return {}

    def get_noise_switch(self):
        """
        This method returns a list of shared theano variables representing switches for values in the model that
        get turned on for training and turned off for testing.
        The variables should be set to either 0. or 1.
        These switch variables are used in theano Switch operations, such as adding noise during training and removing
        it during testing.
        For a usage example, see the BasicLayer in opendeep.models.single_layer.basic package.

        Returns
        -------
        list
            List of SharedVariable used to set the Switches. Defaults to an empty list.
        """
        return []

    #######################################
    # Methods to do with model parameters #
    #######################################
    def get_params(self):
        """
        This returns the list of theano shared variables that will be trained by the :class:`Optimizer`.
        These parameters are used in the gradient.

        Returns
        -------
        list(SharedVariable)
            Flattened list of theano shared variables to be trained with an :class:`Optimizer`. These are the
            parameters for the model.

        Raises
        ------
        NotImplementedError
            If the function hasn't been implemented for the specific model.
        """
        log.critical("%s does not have a get_params function!", str(type(self)))
        raise NotImplementedError("Please implement a get_params method for %s" % str(type(self)))

    def get_param_values(self, borrow=True):
        """
        This returns a list of the parameter values for the model.
        This method is useful when you want to save the model parameters, or are doing distributed programming
        and want to train parallel models.

        Parameters
        ----------
        borrow : bool, optional
            Theano 'borrow' parameter for get_value() method on shared variables. Defaults to True.

        Returns
        -------
        list(array_like)
            List of theano/numpy arrays of values for the model parameters.

        Raises
        ------
        NotImplementedError
            If `get_params()` hasn't been implemented.
        AttributeError
            If a parameter isn't a SharedVariable.
        """
        # try to use theano's get_value() on each parameter returned by get_params()
        try:
            params = get_shared_values(self.get_params(), borrow=borrow)
        except NotImplementedError:
            log.exception("%s cannot get parameters, is missing get_params() method!", str(type(self)))
            raise
        except AttributeError as e:
            log.exception("%s cannot get parameters, there was an AttributeError %s "
                          "when going through the get_params()",
                          str(type(self)), str(e))
            raise

        return params

    def set_param_values(self, param_values, borrow=True):
        """
        This sets the model parameters from the list of values given.
        This method is useful when you are loading model parameters, or are doing distributed programming and
        want to train parallel models.

        The order of param_values matters! It must be the same as the order of parameters returned from
        `self.get_params()`!

        Parameters
        ----------
        param_values : list(array_like)
            List of theano/numpy arrays of values to use for the model parameters.
        borrow : bool
            Theano 'borrow' parameter for `set_value()` method on shared variables.

        Returns
        -------
        bool
            Whether or not successfully set parameters.
        """
        params = self.get_params()

        # make sure the input list of values is the same length as the params for the model.
        if len(param_values) != len(params):
            log.error("%s length of input params to set_param_values() different from length of self.get_params(). "
                      "Input was %s, expected %s",
                      str(type(self)), str(len(param_values)), str(len(self.get_params())))
            return False

        # for each parameter and value in order, set the value!
        try:
            set_shared_values(params, param_values, borrow=borrow)
        except Exception as e:
            log.exception("%s had Exception %s",
                          str(type(self)), str(e))
            return False

        return True

    def save_params(self, param_file):
        """
        This saves the model's parameters (pickles them) to the `param_file` (pickle file)

        Parameters
        ----------
        param_file : str
            Filename of pickled params file to save to.

        Returns
        -------
        bool
            Whether or not successfully saved the file.
        """
        # make sure outdir was not set to false (no saving or outputs)
        if hasattr(self, 'outdir') and self.outdir:
            # By default, try to dump all the values from get_param_values into a pickle file.
            params = self.get_param_values()

            param_path = os.path.join(self.outdir, param_file)
            param_file = os.path.realpath(param_path)

            # force extension to be .pkl if it isn't a pickle file
            _, extension = os.path.splitext(param_file)
            if extension.lower() != ".pkl" or extension.lower() != ".pickle" or extension.lower() != ".p":
                ''.join([param_file, '.pkl'])

            log.debug('Saving %s parameters to %s',
                      str(type(self)), str(param_file))
            # try to dump the param values
            with open(param_file, 'wb') as f:
                try:
                    pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    log.exception("Some issue saving model %s parameters to %s! Exception: %s",
                                  str(type(self)), str(param_file), str(e))
                    return False
                finally:
                    f.close()

            return True
        else:
            return False

    def load_params(self, param_file):
        """
        This loads the model's parameters from the param_file (pickle file)

        Parameters
        ----------
        param_file : str
            Filename of pickled params file (the file holding the pickled model parameters).

        Returns
        -------
        bool
            Whether or not successfully loaded parameters.
        """
        param_file = os.path.realpath(param_file)

        # make sure it is a pickle file
        ftype = file_ops.get_file_type(param_file)
        if ftype == file_ops.PKL:
            log.debug("loading model %s parameters from %s",
                      str(type(self)), str(param_file))
            # try to grab the pickled params from the specified param_file path
            with open(param_file, 'r') as f:
                loaded_params = pickle.load(f)
            self.set_param_values(loaded_params)
            return True
        # if get_file_type didn't return pkl or none, it wasn't a pickle file
        elif ftype:
            log.error("Param file %s doesn't have a supported pickle extension!", str(param_file))
            return False
        # if get_file_type returned none, it couldn't find the file
        else:
            log.error("Param file %s couldn't be found!", str(param_file))
            return False

    def save_args(self, args_file="config.pkl"):
        """
        This saves the model's initial configuration parameters (`self.args`) in a pickle file.

        Parameters
        ----------
        args_file : str, optional
            Filename of pickled configuration parameters. Defaults to 'config.pkl'.

        Returns
        -------
        bool
            Whether or not successfully saved the file.
        """
        # make sure outdir is not set to False (no outputs/saving)
        if hasattr(self, 'outdir') and self.outdir:
            args_path = os.path.join(self.outdir, args_file)
            args_file = os.path.realpath(args_path)

            # force extension to be .pkl if it isn't a pickle file
            _, extension = os.path.splitext(args_file)
            if extension.lower() != ".pkl" or extension.lower() != ".pickle" or extension.lower() != ".p":
                ''.join([args_file, '.pkl'])

            log.debug('Saving %s configuration to %s',
                      str(type(self)), str(args_file))
            # try to dump the param values
            with open(args_file, 'wb') as f:
                try:
                    pickle.dump(self.args, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    log.exception("Some issue saving model %s parameters to %s! Exception: %s",
                                  str(type(self)), str(args_file), str(e))
                    return False
                finally:
                    f.close()

            return True
        else:
            return False