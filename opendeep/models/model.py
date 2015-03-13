"""
.. module:: model

This module defines the generic Model class - which represents everything from a single layer to a full-blown deep network.
Models are the reusable, modular building blocks for the deep networks.
"""

__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import logging
import os
import cPickle
# internal references
from opendeep.utils.config import combine_config_and_defaults
from opendeep.utils import file_ops
from opendeep.utils.misc import set_shared_values, get_shared_values

log = logging.getLogger(__name__)

class Model(object):
    """
    The :class:`Model` is a generic class for everything from a single layer to complex multi-layer behemoths (which can be a
    combination of multiple models linked through input_hooks and hidden_hooks).

    Think of a Model like Legos - you can attach single pieces together as well as multi-piece units together. The main vision
    of OpenDeep is to provide a lightweight, highly modular structure that makes creating and experimenting with new models
    as easy as possible. Much of current deep learning progress has come from combining multiple deep models together for complex
    tasks - such as the image captioning system with convolutional networks + recurrent networks.

    When creating Theano functions inside of models, use the opendeep.function wrapper instead of the basic theano.function -
    this changes unused inputs from an error to a warning. Most likely, unused inputs shouldn't be a breaking error.
    """

    def __init__(self, config=None, defaults=None, inputs_hook=None, hiddens_hook=None, params_hook=None):
        """
        This creates the model's combined configuration params from config and defaults into a self.args dictionary-like
        object (meaning it implements collections.Mapping and you can use self.args.get('parameter') to access something).

        Further, your model implementations should accept optional inputs_hook and hiddens_hook (if applicable) to set your
        inputs and hidden representation in a modular fashion, allowing models to link together.
        inputs_hook is a tuple of (shape, variable) that should replace the default model inputs.
        hiddens_hook is a tuple of (shape, variable) that should replace the default model hidden representation
        (which means you need to adapt creating your computation graph to not care about the inputs and to instead
        compute outputs directly from the hidden variable provided).
        You can also accept a params_hook to share model parameters rather than instantiate a new set of parameters.
        ------------------

        :param config: A dictionary-like object containing all the necessary user-defined parameters for the model.
        This means it either implements collections.Mapping or is a file path to a JSON or YAML configuration file.
        :type config: collections.Mapping object or String (.json file path or .yaml file path)

        :param defaults: A dictionary-like object containing all the necessary default parameters for the model.
        This means it either implements collections.Mapping or is a file path to a JSON or YAML configuration file.
        :type defaults: collections.Mapping object or String (.json file path or .yaml file path)

        :param inputs_hook: Routing information for the model to accept inputs from elsewhere. This is used for linking
        different models together (e.g. setting the Sigmoid model's input layer to the DAE's hidden layer gives a
        newly supervised classification model). For now, you need to include the shape information (normally the
        dimensionality of the input i.e. n_in).
        :type inputs_hook: Tuple of (shape, variable)

        :param hiddens_hook: Routing information for the model to accept its hidden representation from elsewhere.
        This is used for linking different models together (e.g. setting the GSN model's hidden layers to the RNN's
        output layer gives the RNN-GSN model, a deep recurrent model.) For now, you need to include the shape information
        (normally the dimensionality of the hiddens i.e. n_hidden).
        :type hiddens_hook: Tuple of (shape, variable)

        :param params_hook: A list of model parameters (shared theano variables) that you should use when constructing
        this model (instead of initializing your own shared variables). This parameter is useful when you want to have
        two versions of the model that use the same parameters - such as a training model with dropout applied to layers
        and one without for testing, where the parameters are shared between the two.
        :type params_hook: List(theano shared variable)
        """
        log.info("Creating a new instance of %s", str(type(self)))

        # set self.args to be the combination of the defaults and the config dictionaries
        self.args = combine_config_and_defaults(config, defaults)

        # log the arguments.
        log.debug("%s self.args from config parameters: %s", str(type(self)), str(self.args))


    ######################################################################
    # Methods for the symbolic inputs, hiddens, and outputs of the model #
    ######################################################################
    def get_inputs(self):
        """
        This should return the input(s) to the model's computation graph. This is called by the Optimizer when creating
        the theano train function on the cost expression returned by get_train_cost().

        This should normally return the same theano variable list that is used in the inputs= argument to the f_predict
        function.
        ------------------

        :return: Theano variables representing the input(s) to the training function.
        :rtype: List(theano variable)
        """
        log.critical("%s does not have a get_inputs function!", str(type(self)))
        raise NotImplementedError("Please implement a get_inputs method for %s" % str(type(self)))


    def get_hiddens(self):
        """
        This method will return the model's hidden representation expression (if applicable) from the computational graph.

        This will also be used for creating hooks to link models together, where these hidden variables can be strung as the inputs or
        hiddens to another model :)
        ------------------

        :return: theano expression of the hidden representation from this model's computation
        :rtype: theano tensor (expression)
        """
        log.critical("%s get_hiddens method not implemented!", str(type(self)))
        raise NotImplementedError("Please implement a get_hiddens method for %s" % str(type(self)))


    def get_outputs(self):
        """
        This method will return the model's output variable expression from the computational graph. This should be what is given for the
        outputs= part of the 'f_predict' function from self.predict().

        This will be used for creating hooks to link models together, where these outputs can be strung as the inputs or hiddens to another
        model :)

        Example: gsn = GSN()
                 softmax = SoftmaxLayer(inputs_hook=gsn.get_outputs())
        ------------------

        :return: theano expression of the outputs from this model's computation
        :rtype: theano tensor (expression)
        """
        log.critical("%s get_outputs method not implemented!", str(type(self)))
        raise NotImplementedError("Please implement a get_outputs method for %s" % str(type(self)))


    #############################################
    # Methods for running the model on an input #
    #############################################
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
        log.critical("%s predict method not implemented!", str(type(self)))
        raise NotImplementedError("Please implement a predict method for %s" % str(type(self)))


    ###########################################################
    # Methods to do with training the model with an Optimizer #
    ###########################################################
    def get_train_cost(self):
        """
        This returns the expression that represents the cost given an input, which is used for the Optimizer during
        training. The reason we can't just compile a f_train theano function is because updates need to be calculated
        for the parameters during gradient descent - and these updates are created in the Optimizer object.
        ------------------

        :return: theano expression of the model's training cost, from which parameter gradients will be computed.
        :rtype: theano tensor
        """
        log.critical("%s does not have a get_train_cost function!", str(type(self)))
        raise NotImplementedError("Please implement a get_train_cost method for %s" % str(type(self)))


    def get_updates(self):
        """
        This should return any theano updates from the model (used for things like random number generators).
        Most often comes from theano's 'scan' op. Check out its documentation at http://deeplearning.net/software/theano/library/scan.html.
        This is used with the optimizer to create the training function - the 'updates=' part of the theano function.
        ------------------

        :return: updates from the theano computation for the model to be used during Optimizer.train() (but not including
        training parameter updates - those are calculated by the Optimizer) These are expressions for new SharedVariable
        values.
        :rtype: (iterable over pairs (shared_variable, new_expression). List, tuple, or dict.)
        """
        # TODO: should we do the parameter decays from get_decay_params() in the model updates? Right now I'm not because it seems less modular
        # by default, assume the model doesn't have updates - it's your job to return them in this method.
        return None


    def get_monitors(self):
        """
        This returns a dictionary of (monitor_name: monitor_function) of variables (monitors) whose values we care
        about during training. For every monitor returned by this method, the function will be run on the train/validation/test
        dataset and its value will be reported.

        Again, please avoid recompiling the monitor functions every time - check your hasattr to see if they already
        exist!
        ------------------

        :return: Dictionary of String: theano_function for each monitor variable we care about in the model.
        :rtype: Dictionary
        """
        log.critical("%s does not have a get_monitors function!", str(type(self)))
        raise NotImplementedError("Please implement a get_monitors method for %s" % str(type(self)))


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
        # no decay parameters by default
        return []


    def get_lr_scalers(self):
        """
        This method lets you scale the overall learning rate in the Optimizer to individual parameters. Returns a dictionary mapping
        model_parameter: learning_rate_scaling_factor. Default is no scaling.
        ------------------

        :return: dictionary mapping the model parameters to their learning rate scaling factor
        :rtype: Dictionary(shared_variable: float)
        """
        # By default, no learning rate scaling.
        return {}


    #######################################
    # Methods to do with model parameters #
    #######################################
    def get_params(self):
        """
        This returns the list of theano shared variables that will be trained by the Optimizer. These parameters are used in the gradient.
        ------------------

        :return: flattened list of theano shared variables to be trained
        :rtype: List(shared_variables)
        """
        log.critical("%s does not have a get_params function!", str(type(self)))
        raise NotImplementedError("Please implement a get_params method for %s" % str(type(self)))


    def get_param_values(self, borrow=True):
        """
        This returns a list of the parameter values for the model.
        This method is useful when you want to save the model parameters, or are doing distributed programming and want to train parallel
        models.
        ------------------

        :param borrow: theano 'borrow' parameter for get_value() method on shared variables
        :type borrow: Boolean

        :return: list of theano/numpy arrays of values for the model parameters
        :rtype: List(array)
        """
        # try to use theano's get_value() on each parameter returned by get_params()
        try:
            params = get_shared_values(self.get_params(), borrow=borrow)
        except NotImplementedError:
            log.exception("%s cannot get parameters, is missing get_params() method!", str(type(self)))
            raise
        except AttributeError as e:
            log.exception("%s cannot get parameters, there was an AttributeError %s when going through the get_params()",
                          str(type(self)), str(e))
            raise

        return params


    def set_param_values(self, param_values, borrow=True):
        """
        This sets the model parameters from the list of values given.
        This method is useful when you are loading model parameters, or are doing distributed programming and want to train parallel models.
        The order of param_values matters! It must be the same as the order of parameters returned from self.get_params()!
        ------------------

        :param param_values: list of theano/numpy arrays of values for the model parameters
        :type param_values: List(array)

        :param borrow: theano 'borrow' parameter for set_value() method on shared variables
        :type borrow: Boolean

        :return: whether or not successful
        :rtype: Boolean
        """
        params = self.get_params()

        # make sure the input list of values is the same length as the params for the model.
        if len(param_values) != len(params):
            log.error("%s length of input params to set_param_values() different from length of self.get_params(). Input was %s, expected %s",
                      str(type(self)), str(len(param_values)), str(len(self.get_params())))
            return False

        # for each parameter and value in order, set the value!
        try:
            set_shared_values(params, param_values, borrow=borrow)
        except Exception, e:
            log.exception("%s had Exception %s",
                          str(type(self)), str(e))
            return False

        return True


    def save_params(self, param_file):
        """
        This saves the model's parameters to the param_file (pickle file)
        ------------------

        :param param_file: filename of pickled params file
        :type param_file: String

        :return: whether or not successful
        :rtype: Boolean
        """
        # By default, try to dump all the values from get_param_values into a pickle file.
        params = self.get_param_values()

        param_file = os.path.realpath(param_file)

        # force extension to be .pkl if it isn't a pickle file
        _, extension = os.path.splitext(param_file)
        if extension.lower() != ".pkl" or extension.lower() != ".pickle" or extension.lower() != ".p":
            ''.join([param_file, '.pkl'])

        log.debug('Saving %s parameters to %s...',
                  str(type(self)), str(param_file))
        # try to dump the param values
        with open(param_file, 'wb') as f:
            try:
                cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)
            except Exception as e:
                log.exception("Some issue saving model %s parameters to %s! Exception: %s",
                              str(type(self)), str(param_file), str(e))
                return False
            finally:
                f.close()

        return True


    def load_params(self, param_file):
        """
        This loads the model's parameters from the param_file (pickle file)
        ------------------

        :param param_file: filename of pickled params file
        :type param_file: String

        :return: whether or not successful
        :rtype: Boolean
        """
        param_file = os.path.realpath(param_file)

        # make sure it is a pickle file
        ftype = file_ops.get_file_type(param_file)
        if ftype == file_ops.PKL:
            log.debug("loading model %s parameters from %s...",
                      str(type(self)), str(param_file))
            # try to grab the pickled params from the specified param_file path
            with open(param_file, 'r') as f:
                loaded_params = cPickle.load(f)
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