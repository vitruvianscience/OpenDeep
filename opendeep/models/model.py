"""
This module defines the generic Model class -
which represents everything from a single layer to a full-blown deep network.

Models are the reusable, modular building blocks for deep networks. Their power comes from
their ability to connect with other Models.
"""
# standard libraries
import logging
import os
import time
# internal references
import opendeep.models
from opendeep.utils.decorators import init_optimizer
from opendeep.utils import file_ops
from opendeep.utils.constructors import function
from opendeep.utils.misc import (make_time_units_string, raise_to_list, add_kwargs_to_dict)
from opendeep.utils.file_ops import mkdir_p

try:
    import cPickle as pickle
except ImportError:
    import pickle

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

hdf5_param_key = "params"
class_key = "class"

log = logging.getLogger(__name__)


class Model(object):
    """
    The :class:`Model` is a generic class for everything from a single layer to complex multi-layer behemoths
    (which can be a combination of multiple models linked through inputs, hiddens, and params).

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
    inputs : list
        List of [tuple(shape, `Theano.TensorType`)] or None describing the inputs to use for this Model.
        `shape` will be a monad tuple representing known sizes for each dimension in the `Theano.TensorType`.
        The length of `shape` should be equal to number of dimensions in `Theano.TensorType`, where the shape
        element is an integer representing the size for its dimension, or None if the shape isn't known.
        For example, if you have a matrix with unknown batch size but fixed feature size of 784, `shape` would
        be: (None, 784). The full form of `inputs` would be:
        [((None, 784), <TensorType(float32, matrix)>)].
    hiddens : list
        List of [int or shape or Tuple of (shape, hiddens_variable)] or None to use as the hidden
        representation for this Model.
    output_size : int or shape tuple
        Describes the shape of the output dimensionality for this Model.
    params : Dict or None
        A dict of string_name: SharedVariable representing the parameters to use for this Model.
    outdir : str
        The filepath to save outputs for this Model (such as pickled parameters created during training,
        visualizations, etc.).
    f_run : function, or attribute doesn't exist if not compiled.
        Theano function for running the model's computation on an input. This gets set during compile_run_fn().
    switches_on : bool or None
        If all the switches from `self.get_switches()` have been turned off (False) or on (True). It will be
        None if we don't know the state of the switches.
    """

    def __init__(self, inputs=None, hiddens=None, outputs=None,
                 params=None,
                 outdir=None,
                 **kwargs):
        """
        Initialize a new Model.

        Your model implementations should accept optional inputs and hiddens Theano symbolic expressions
        or variables (if applicable) to set your inputs and hidden representation in a modular fashion,
        allowing models to link together. `inputs` can have a tuple of (shape, variable) that should replace
        the default model inputs. hiddens can have a tuple of (shape, variable) that should replace the
        default model hidden representation (which means you need to adapt creating your computation graph
        to not care about the inputs and to instead run outputs directly from the hidden variable provided).
        You can also accept a params to share model parameters rather than instantiate a new set of parameters.

        Parameters
        ----------
        inputs : List of [tuple(shape, `Theano.TensorType`) or Model] or None
            The dimensionality of the inputs for this model, and the routing information for the model
            to accept inputs from elsewhere. This is used for linking
            different models together (e.g. setting the Softmax model's input layer to the DAE's hidden layer gives a
            newly supervised classification model). `shape` will be a monad tuple representing known
            sizes for each dimension in the `Theano.TensorType`. The length of `shape` should be equal to number of
            dimensions in `Theano.TensorType`, where the shape element is an integer representing the size for its
            dimension, or None if the shape isn't known. For example, if you have a matrix with unknown batch size
            but fixed feature size of 784, `shape` would be: (None, 784). The full form of `inputs` would be:
            [((None, 784), <TensorType(float32, matrix)>)]. If a :class:`Model` is given as the input, it replaces
            the tuple with zip(Model.output_size, Model.get_outputs()).
        hiddens : List of [tuple(shape, `Theano.TensorType`) or shape] or None, optional
            The dimensionality of the hidden representation for this model, and/or the routing information for
            the model to accept its hidden representation from elsewhere.
            This is used for linking different models together (e.g. setting the GSN model's hidden layers to the RNN's
            output layer gives the RNN-GSN model, a deep recurrent model.) For now, variable hook tuples need to
            include the shape information (normally the dimensionality of the hiddens i.e. n_hidden). This shape
            information is the same format as the monad for `inputs`.
        outputs : List of [int or shape tuple], optional
            The dimensionality of the output(s) for this model. Shape here is the shape monad described in `inputs`.
        params : Dict(string_name: theano SharedVariable), optional
            A dictionary of model parameters (shared theano variables) that you should use when constructing
            this model (instead of initializing your own shared variables). This parameter is useful when you want to
            have two versions of the model that use the same parameters - such as siamese networks or pretraining some
            weights.
        outdir : str, optional
            The directory you want outputs (parameters, images, etc.) to save to. If None, nothing will
            be saved.
        kwargs : dict, optional
            This will be all the other left-over keyword parameters passed to the class as a
            dictionary of {param: value}. These get created into `self.args` along with outdir and outputs.
        """
        self._classname = self.__class__.__name__
        log.info("Creating a new instance of %s", self._classname)

        # Necessary inputs to a Model - these are the minimum requirements for modularity to work.
        self.inputs = raise_to_list(inputs)
        if self.inputs is not None:
            ins = []
            # deal with Models or ModifyLayers being passed as an input.
            for input in self.inputs:
                if hasattr(input, 'output_size') and hasattr(input, 'get_outputs'):
                    sizes = raise_to_list(input.output_size)
                    outs = raise_to_list(input.get_outputs())
                    if len(sizes) == 1 and len(sizes) < len(outs):
                        sizes = sizes * len(outs)
                    input = raise_to_list(zip(sizes, outs))
                    for i in input:
                        ins.append(i)
                else:
                    ins.append(input)
            # replace self.inputs
            self.inputs = ins

        self.hiddens = raise_to_list(hiddens)
        self.output_size = raise_to_list(kwargs.get('output_size', outputs))
        self.params = params or {}
        self.outdir = outdir

        # make the directory to output configuration and parameters from the model
        if self.outdir:
            self.outdir = os.path.realpath(self.outdir)
            mkdir_p(self.outdir)

        # copy all of the parameters from the class into an args (configuration) dictionary
        self.args = {}
        self.args = add_kwargs_to_dict(kwargs.copy(), self.args)

        self.args['inputs'] = self.inputs
        self.args['hiddens'] = self.hiddens
        if self.output_size is not None:
            self.args['output_size'] = self.output_size
        self.args['params'] = self.params
        self.args['outdir'] = self.outdir

        # log the arguments.
        log.info("%s self.args: %s", self._classname, str(self.args))
        # save the arguments.
        self.save_args()
        # Boom! Hyperparameters are now dealt with. Take that!

        # Don't know the position of switches!
        self.switches_on = None

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
        log.critical("%s does not have a get_inputs function!", self._classname)
        raise NotImplementedError("Please implement a get_inputs method for %s" % self._classname)

    def get_hiddens(self):
        """
        This method will return the model's hidden representation expression (if applicable)
        from the computational graph. This is normally useful for unsupervised models, whose hidden units
        learn the representation of the input.

        This will also be used for creating hooks to link models together, where these hidden variables can be strung
        as the inputs or hiddens to another model :)

        Returns
        -------
        theano expression or list(theano expression)
            Theano expression(s) of the hidden representation from this model's computation.

        Raises
        ------
        NotImplementedError
            If the function hasn't been implemented for the specific model.
        """
        log.critical("%s get_hiddens method not implemented!", self._classname)
        raise NotImplementedError("Please implement a get_hiddens method for %s" % self._classname)

    def get_outputs(self):
        """
        This method will return the model's output variable expression from the computational graph.
        This should be what is given for the outputs= part of the 'f_run' function from `self.run()`.

        This will be used for creating hooks to link models together,
        where these outputs can be strung as the inputs or hiddens to another model :)

        Returns
        -------
        theano expression or list(theano expression)
            Theano expression(s) of the outputs from this model's computation graph.

        Examples
        --------
        Here is an example showing the `get_outputs()` method in the GSN model used in an `inputs` hook
        to a Softmax model::

            from opendeep.models import GSN, Softmax
            gsn = GSN(inputs=28*28, hiddens=1000, layers=2, walkbacks=4)
            softmax = Softmax(inputs=zip(gsn.output_size, gsn.get_outputs()), outputs=10)

        Raises
        ------
        NotImplementedError
            If the function hasn't been implemented for the specific model.
        """
        log.critical("%s get_outputs method not implemented!", self._classname)
        raise NotImplementedError("Please implement a get_outputs method for %s" % self._classname)

    ##########################################################
    # Methods for running and training the model on an input #
    ##########################################################
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

        Returns
        -------
        Theano function
            The compiled theano function for running the model.
        """
        if not getattr(self, 'f_run', None):
            log.debug("Compiling f_run...")
            t = time.time()
            self.f_run = function(inputs  = raise_to_list(self.get_inputs()),
                                  outputs = self.get_outputs(),
                                  updates = self.get_updates(),
                                  name    = 'f_run')
            log.debug("Compilation done. Took %s", make_time_units_string(time.time() - t))
        else:
            log.debug('f_run already exists!')

        return self.f_run

    def run(self, input):
        """
        This method will return the model's output (run through the function), given an input. In the case that
        input_hooks or hidden_hooks are used, the function should use them appropriately and assume they are the input.

        .. note::
            If the Model doesn't have an `f_run` attribute,
            it will run `compile_run_fn()` to compile the appropriate function.

        Parameters
        ----------
        input : tensor or list(tensor)
            Theano/numpy tensor-like object(s) that is the input(s) into the model's computation graph.

        Returns
        -------
        array_like or list(array_like)
            Array_like object that is the output(s) of the model's computation graph run on the given input(s).
        """
        # set the noise switches off for running (this only happens the first time)!
        old_switch_vals = []
        if self.switches_on is not False:
            old_switch_vals = [switch.get_value() for switch in raise_to_list(self.get_switches())]
            self.turn_off_switches()

        # check if the run function is already compiled, otherwise compile it!
        if not getattr(self, 'f_run', None):
            self.compile_run_fn()

        # because we use the splat to account for multiple inputs to the function, make sure input is a list.
        input = raise_to_list(input)
        # return the results of the run function!
        output = self.f_run(*input)

        # reset any switches to how they were!
        if len(old_switch_vals) > 0:
            self.set_switches(old_switch_vals)

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
        list(tensor_like)
            The list of generated values from the Model (starting from `initial` if applicable).

        Raises
        ------
        NotImplementedError
            If the function hasn't been implemented for the specific model.
        """
        log.exception("Generate method not implemented for Model %s", self._classname)
        raise NotImplementedError("Generate method not implemented for Model %s" % self._classname)

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
        # by default, assume the model doesn't have updates - it's your job to return them in this method.
        return None

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

    def get_switches(self):
        """
        This method returns a list of shared theano variables representing switches for values in the model that
        get turned on or off for training/testing.
        The variables should be set to either 0. or 1.
        These switch variables are used in theano Switch operations, such as adding noise during training and removing
        it during testing.

        Returns
        -------
        list
            List of SharedVariable used to set the Switches. Defaults to an empty list.
        """
        return []

    def flip_switches(self):
        """
        This helper method flips all Theano switches specified by `get_switches()` to 0. or 1. (the opposite value
        that the switch is currently set to).
        """
        switches = raise_to_list(self.get_switches())
        if len(switches) > 0:
            log.debug("Flipping %d switches for %s!" % (len(switches), self._classname))
            [switch.set_value(1. - switch.get_value()) for switch in switches]
            if self.switches_on is not None:
                self.switches_on = not self.switches_on

    def turn_off_switches(self):
        """
        This helper method turns all Theano switches by `get_switches()` to their off position of 0./False
        """
        switches = raise_to_list(self.get_switches())
        if len(switches) > 0:
            log.debug("Turning off %d switches for %s!" % (len(switches), self._classname))
            [switch.set_value(0.) for switch in switches]
            self.switches_on = False

    def turn_on_switches(self):
        """
        This helper method turns all Theano switches by `get_switches()` to their on position of 1./True
        """
        switches = raise_to_list(self.get_switches())
        if len(switches) > 0:
            log.debug("Turning on %d switches for %s!" % (len(switches), self._classname))
            [switch.set_value(1.) for switch in switches]
            self.switches_on = True

    def set_switches(self, values):
        """
        This helper method sets all Theano switches from `get_switches()` to the `values` parameter specified.

        Parameters
        ----------
        values : list(boolean)
        """
        switches = raise_to_list(self.get_switches())
        values = raise_to_list(values)
        values = [1. if val else 0. for val in values]
        assert len(switches) == len(values), "Switches (len %d) needs to be same length as values (len %d)!" % \
                                             (len(switches), len(values))
        log.debug("Setting specified values for %d switches!" % len(switches))
        [switch.set_value(val) for switch, val in zip(switches, values)]
        self.switches_on = None

    def get_loss(self):
        """
        Helper function for defining model-specific loss functions. Normally, you would pass an instance of
        :class:`opendeep.optimization.loss.Loss` to the optimizer. However, sometimes models or layers have
        specific, fixed loss functions that need to be implemented internally. If that is the case, implement
        this function.

        Returns
        -------
        theano_expression or tuple(list(theano_variable), theano_expression) or None
            The loss expression, or a tuple containing the theano variables (i.e. matrix, tensor3, etc.)
            used as targets when calculating loss and the theano expression representing the loss function.
        """
        return None

    @init_optimizer
    def train(self, optimizer, **kwargs):
        """
        This is a syntactic sugar method for training the model with a given Optimizer.
        See train() in Optimizer for parameters.
        """
        optimizer.train(**kwargs)

    #######################################
    # Methods to do with model parameters #
    #######################################
    def get_params(self):
        """
        This returns the ordered dictionary of {string_name: theano shared variables} that will be trained by
        the :class:`Optimizer`. These parameters are used in the gradient.

        Returns
        -------
        OrderedDict(str: SharedVariable)
            Ordered dictionary of {string_name: theano shared variables} to be trained with an :class:`Optimizer`.
            These are the parameters for the model.

        Raises
        ------
        NotImplementedError
            If the function hasn't been implemented for the specific model.
        """
        log.critical("%s does not have a get_params function!", self._classname)
        raise NotImplementedError("Please implement a get_params method for %s" % self._classname)

    def get_param_values(self, borrow=False):
        """
        This returns a dictionary of the parameter values for the model.
        This method is useful when you want to save the model parameters, or are doing distributed programming
        and want to train parallel models.

        Parameters
        ----------
        borrow : bool, optional
            Theano 'borrow' parameter for get_value() method on shared variables.

        Returns
        -------
        dict(str: array_like)
            Dict of {string_name: theano/numpy arrays} of values for the model parameters.

        Raises
        ------
        NotImplementedError
            If `get_params()` hasn't been implemented.
        AttributeError
            If a parameter isn't a SharedVariable.
        """
        # try to use theano's get_value() on each parameter returned by get_params()
        try:
            params = {name: variable.get_value(borrow=borrow) for name, variable in self.get_params().items()}
        except NotImplementedError:
            log.exception("%s cannot get parameters, is missing get_params() method!", self._classname)
            raise
        except AttributeError as e:
            log.exception("%s cannot get parameters, there was an AttributeError %s "
                          "when going through the get_params()",
                          self._classname, str(e))
            raise

        return params

    def set_param_values(self, param_values, borrow=False):
        """
        This sets the model parameters from the dictionary of values given.
        This method is useful when you are loading model parameters, or are doing distributed programming and
        want to train parallel models.

        Parameters
        ----------
        param_values : dict(string: array_like)
            Dict of {string_name: theano/numpy arrays} of values to use for the model parameters.
        borrow : bool
            Theano 'borrow' parameter for `set_value()` method on shared variables.

        Returns
        -------
        bool
            Whether or not successfully set parameters.
        """
        params = self.get_params()

        # Find any differences in keys supplied vs keys of model params
        param_values_keyset = set(param_values.keys())
        params_keyset = set(params.keys())
        intersection = param_values_keyset.intersection(params_keyset)
        for key in param_values_keyset - intersection:
            log.warning("Param value was supplied for param %s but it does not exist in the model %s params %s." %
                        (str(key), self._classname, str(params_keyset)))
        for key in params_keyset - intersection:
            log.warning("Param value was supplied for param %s but it does not exist in the model %s params %s." %
                        (str(key), self._classname, str(params_keyset)))

        # for each parameter and value, set the value!
        try:
            for key in intersection:
                params[key].set_value(param_values[key], borrow=borrow)
        except Exception as e:
            log.exception("%s had Exception %s",
                          self._classname, str(e))
            return False

        return True

    def save_params(self, param_file, use_hdf5=False):
        """
        This saves the model's parameters (HDF5 file or pickles them) to the `param_file`.

        Parameters
        ----------
        param_file : str
            Filename of HDF5 or pickled params file to save to.
        use_hdf5 : bool
            Whether to use an HDF5 file for the saved parameters (if h5py is installed).
            Otherwise, it will use pickle.

        Returns
        -------
        bool
            Whether or not successfully saved the file.
        """
        # make sure outdir was not set to false (no saving or outputs)
        if getattr(self, 'outdir', None):
            param_path = os.path.join(self.outdir, param_file)
            param_file = os.path.realpath(param_path)

            ftype = file_ops.get_extension_type(param_file)

            params_dict = self.get_param_values(borrow=False)

            if HAS_H5PY and use_hdf5:
                # force extension to be .hdf5
                if ftype != file_ops.HDF5:
                    param_file = ''.join([param_file, '.hdf5'])

                log.debug('Saving %s parameters to %s',
                          self._classname, str(param_file))

                # try to dump the param values
                f = h5py.File(param_file, 'w')
                try:
                    if hdf5_param_key not in f:
                        param_group = f.create_group(hdf5_param_key)
                    else:
                        param_group = f[hdf5_param_key]
                    for name, param in params_dict.items():
                        if name in param_group:
                            dset = param_group[name]
                            dset[...] = param
                        else:
                            dset = param_group.create_dataset(name, data=param)
                    f.flush()
                except Exception as e:
                    log.exception("Some issue saving model %s parameters to %s! Exception: %s",
                                  self._classname, str(param_file), str(e))
                    return False
                finally:
                    f.close()
            else:
                # force extension to be .pkl if it isn't a pickle file
                if ftype != file_ops.PKL:
                    param_file = ''.join([param_file, '.pkl'])

                log.debug('Saving %s parameters to %s',
                          self._classname, str(param_file))

                # try to dump the param values
                with open(param_file, 'wb') as f:
                    try:
                        pickle.dump(params_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                    except Exception as e:
                        log.exception("Some issue saving model %s parameters to %s! Exception: %s",
                                      self._classname, str(param_file), str(e))
                        return False
                    finally:
                        f.close()
            # all done
            return True
        else:
            return False

    def load_params(self, param_file):
        """
        This loads the model's parameters from the param_file (hdf5 or pickle file)

        Parameters
        ----------
        param_file : str
            Filename of hdf5 or pickled params file (the file holding the model parameters).

        Returns
        -------
        bool
            Whether or not successfully loaded parameters.
        """
        param_file = os.path.realpath(param_file)

        # make sure it is a pickle file
        ftype = file_ops.get_file_type(param_file)

        log.debug("loading %s model parameters from %s",
                  self._classname, str(param_file))

        if ftype == file_ops.PKL:
            # try to grab the pickled params from the specified param_file path
            with open(param_file, 'rb') as f:
                loaded_params = pickle.load(f)
            self.set_param_values(loaded_params, borrow=False)
            return True

        elif ftype == file_ops.HDF5:
            if HAS_H5PY:
                f = h5py.File(param_file)
                try:
                    params = f[hdf5_param_key]
                    self.set_param_values(params)
                except Exception as e:
                    log.exception("Some issue loading model %s parameters from %s! Exception: %s",
                                  self._classname, str(param_file), str(e))
                    return False
                finally:
                    f.close()
            else:
                log.error("Please install the h5py package to read HDF5 files!")
                return False
        # if get_file_type didn't return pkl, hdf5, or none
        elif ftype:
            log.error("Param file %s doesn't have a supported pickle or HDF5 extension!", str(param_file))
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
        if getattr(self, 'outdir', None):
            args_path = os.path.join(self.outdir, args_file)
            args_file = os.path.realpath(args_path)

            ftype = file_ops.get_extension_type(args_file)

            args = self.args.copy()
            args[class_key] = self._classname

            # force extension to be .pkl if it isn't a pickle file
            if ftype != file_ops.PKL:
                args_file = ''.join([args_file, '.pkl'])
            log.debug('Saving %s configuration to %s',
                      self._classname, str(args_file))
            # try to dump the args values
            with open(args_file, 'wb') as f:
                try:
                    pickle.dump(args, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    log.exception("Some issue saving model %s configuration to %s! Exception: %s",
                                  self._classname, str(args_file), str(e))
                    return False
                finally:
                    f.close()

            return True
        else:
            return False

    def save_run(self, filename):
        """
        Saves (pickle) the compiled theano function for running the model.

        Parameters
        ----------
        filename : str
            Filepath to save the compiled run function

        Returns
        -------
        tuple(bool, str)
            Tuple of [whether or not successful] and [complete filepath to saved file].
        """
        # make sure outdir is not set to False (no outputs/saving)
        if getattr(self, 'outdir', None):
            filepath = os.path.join(self.outdir, filename)
            save_file = os.path.realpath(filepath)

            ftype = file_ops.get_extension_type(save_file)

            # force extension to be .pkl if it isn't a pickle file
            if ftype != file_ops.PKL:
                save_file = ''.join([save_file, '.pkl'])

            log.debug('Saving %s compiled run function to %s',
                      self._classname, str(save_file))

            # try to dump the param values
            with open(save_file, 'wb') as f:
                try:
                    run_fn = self.compile_run_fn()
                    pickle.dump(run_fn, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    if "maximum recursion depth exceeded" in str(e):
                        recursion_limit = 50000
                        import sys
                        while "maximum recursion depth exceeded" in str(e):
                            log.debug("found recursion depth bug when pickling function...bumping limit to %d"
                                  % recursion_limit)
                            sys.setrecursionlimit(recursion_limit)
                            try:
                                run_fn = self.compile_run_fn()
                                pickle.dump(run_fn, f, protocol=pickle.HIGHEST_PROTOCOL)
                                return (True, save_file)
                            except Exception as e:
                                if "maximum recursion depth exceeded" not in str(e):
                                    log.exception("Some issue saving model %s run function to %s! Exception: %s",
                                                  self._classname, str(save_file), str(e))
                                    return (False, save_file)
                                recursion_limit += 10000
                    else:
                        log.exception("Some issue saving model %s run function to %s! Exception: %s",
                                          self._classname, str(save_file), str(e))
                        return (False, save_file)
                finally:
                    f.close()
            # all done
            return (True, save_file)
        else:
            return (False, None)

    def copy(self, **kwargs):
        """
        Returns a new copy of this model - same class as self initialized with the args from self.args updated
        with the keyword arguments kwargs supplied.

        Parameters
        ----------
        kwargs : keyword arguments
            Any arguments you want to override during the initialization of the :class:`Model`.

        Returns
        -------
        :class:`Model`
            A copy of the current model (same configurations except those overridden by kwargs).
        """
        args = self.args.copy()
        args.update(kwargs)
        return type(self)(**args)

    def save(self, config_file, param_file, use_hdf5=False):
        """
        Saves this model (and its current parameters) to files.

        Parameters
        ----------
        config_file : str
            Filename of pickled configuration file.
        param_file : str or None
            Filename of hdf5 or pickle file holding the model parameters (in a separate file from `config_file`). If
            None, params will not be saved.
        use_hdf5 : bool
            Whether to use an HDF5 file for the saved parameters (if h5py is installed).
            Otherwise, it will use pickle with separate files.
        """
        # make sure outdir is not set to False (no outputs/saving)
        if getattr(self, 'outdir', None):
            self.save_args(args_file=config_file)
            if param_file is not None:
                self.save_params(param_file=param_file, use_hdf5=use_hdf5)
            return True
        else:
            return False

    @staticmethod
    def load(config_file, param_file=None):
        """
        Returns a new Model from a configuration file.

        Parameters
        ----------
        config_file : str
            Filename of pickled configuration file.
        param_file : str, optional
            Filename of hdf5 or pickle file holding the model parameters (in a separate file from `config_file`
            if you want to load some starting parameters).

        Returns
        -------
        :class:`Model`
            A `Model` instance from the configuration and optionally loaded parameters.
        """
        config_file = os.path.realpath(config_file)

        ftype = file_ops.get_file_type(config_file)

        # deal with pickle
        if ftype == file_ops.PKL:
            log.debug("loading model from %s",
                      str(config_file))
            with open(config_file, 'rb') as f:
                loaded_config = pickle.load(f)
        # if get_file_type didn't return pkl, or none
        elif ftype:
            log.exception("Config file %s doesn't have a supported pickle extension!", str(config_file))
            raise AssertionError("Config file %s doesn't have a supported pickle extension!", str(config_file))
        # if get_file_type returned none, it couldn't find the file
        else:
            log.exception("Config file %s couldn't be found!", str(config_file))
            raise AssertionError("Config file %s couldn't be found!", str(config_file))

        classname = loaded_config.pop(class_key)
        class_ = getattr(opendeep.models, classname)
        model = class_(**loaded_config)
        if param_file is not None:
            model.load_params(param_file=param_file)

        return model
