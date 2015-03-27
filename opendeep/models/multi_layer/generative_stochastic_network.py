'''
.. module:: generative_stochastic_network

This module gives an implementation of the Generative Stochastic Network model.

Based on code from Li Yao (University of Montreal)
https://github.com/yaoli/GSN

This class's main() method by default produces the model trained on MNIST discussed in the paper:
'Deep Generative Stochastic Networks Trainable by Backprop'
Yoshua Bengio, Eric Thibodeau-Laufer
http://arxiv.org/abs/1306.1091

Scheduled noise is added as discussed in the paper:
'Scheduled denoising autoencoders'
Krzysztof J. Geras, Charles Sutton
http://arxiv.org/abs/1406.3269

TODO:
Multimodal transition operator (using NADE) discussed in:
'Multimodal Transitions for Generative Stochastic Networks'
Sherjil Ozair, Li Yao, Yoshua Bengio
http://arxiv.org/abs/1312.5578
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger", "Li Yao"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import os
import time
import logging
# third-party libraries
import numpy
import numpy.random as rng
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG
from theano.compat.python2x import OrderedDict
import PIL
# internal references
from opendeep import cast32, function, sharedX
from opendeep.models.model import Model
from opendeep.utils import file_ops
from opendeep.utils.decay import get_decay_function
from opendeep.utils.activation import get_activation_function
from opendeep.utils.cost import get_cost_function
from opendeep.utils.misc import closest_to_square_factors, make_time_units_string
from opendeep.utils.nnet import get_weights_uniform, get_weights_gaussian, get_bias
from opendeep.utils.noise import salt_and_pepper, add_gaussian
from opendeep.utils.image import tile_raster_images

log = logging.getLogger(__name__)

# Default values to use for some GSN parameters.
# These defaults are used to produce the MNIST results given in the comments top of file.
_defaults = {# gsn parameters
            "layers": 3,  # number of hidden layers to use
            "walkbacks": 5,  # number of walkbacks (generally 2*layers) - need enough to propagate to visible layer
            "input_size": None,  # number of input units - please specify for your dataset!
            "hidden_size": 1500,  # number of hidden units in each layer
            "visible_activation": 'sigmoid',  # activation for visible layer - make appropriate for input data type.
            "hidden_activation": 'tanh',  # activation for hidden layers
            "input_sampling": True,  # whether to sample at each walkback step - makes it like Gibbs sampling.
            "MRG": RNG_MRG.MRG_RandomStreams(1),  # default random number generator from Theano
            "weights_init": "uniform",  # how to initialize weights
            'weights_interval': 'montreal',  # if the weights_init was 'uniform', how to initialize from uniform
            'weights_mean': 0,  # mean for gaussian weights init
            'weights_std': 0.005,  # standard deviation for gaussian weights init
            'bias_init': 0.0,  # how to initialize the bias parameter
            # train param
            "cost_function": 'binary_crossentropy',  # the cost function for training; make appropriate for input type.
            # noise parameters
            "noise_decay": 'exponential',  # noise schedule algorithm
            "noise_annealing": 1.0,  # no noise schedule by default
            "add_noise": True,  # whether to add noise throughout the network's hidden layers
            "noiseless_h1": True,  # whether to keep the first hidden layer uncorrupted
            "hidden_add_noise_sigma": 2,  # sigma value for adding the gaussian hidden layer noise
            "input_salt_and_pepper": 0.4,  # the salt and pepper value for inputs corruption
            # data parameters
            "output_path": 'outputs/gsn/',  # base directory to output various files
            "is_image": True,  # whether the input should be treated as an image
            "vis_init": False}

class GSN(Model):
    '''
    Class for creating a new Generative Stochastic Network (GSN)
    '''
    def __init__(self, config=None, defaults=_defaults, inputs_hook=None, hiddens_hook=None, dataset=None,
                 layers=None, walkbacks=None, input_size=None, hidden_size=None, visible_activation=None,
                 hidden_activation=None, input_sampling=None, MRG=None, weights_init=None, weights_interval=None,
                 weights_mean=None, weights_std=None, bias_init=None, cost_function=None, noise_decay=None,
                 noise_annealing=None, add_noise=None, noiseless_h1=None, hidden_add_noise_sigma=None,
                 input_salt_and_pepper=None, output_path=None, is_image=None, vis_init=None):
        # set up base path for the outputs of the model during training, etc.
        self.outdir = output_path
        if not self.outdir and config:
            self.outdir = config.get('output_path', defaults.get('output_path'))
        else:
            self.outdir = defaults.get('output_path')
        if self.outdir[-1] != '/':
            self.outdir = self.outdir+'/'
        file_ops.mkdir_p(self.outdir)

        # combine everything by passing to Model's init
        super(GSN, self).__init__(**{arg: val for (arg, val) in locals().iteritems() if arg is not 'self'})
        # configs can now be accessed through self! huzzah!

        # variables from the dataset that are used for initialization and image reconstruction
        if self.inputs_hook is None:
            if self.dataset is None:
                # if no inputs_hook or dataset given, look for the dimensionality of the input from the
                # 'input_size' parameter.
                self.N_input = self.input_size
                if self.input_size is None:
                    log.critical("Please either specify input_size in the arguments or provide an example "
                                 "dataset object for input dimensionality.")
                    raise AssertionError("Please either specify input_size in the arguments or provide an example "
                                         "dataset object for input dimensionality.")
            else:
                # otherwise grab shape from dataset example that was given
                self.N_input = self.dataset.get_example_shape()
        else:
            # otherwise grab shape from inputs_hook that was given
            self.N_input = self.inputs_hook[0]

        # if the input should be thought of as an image, either use the specified width and height,
        # or try to make as square as possible.
        if self.is_image:
            (_h, _w) = closest_to_square_factors(self.N_input)
            self.image_width  = self.args.get('width', _w)
            self.image_height = self.args.get('height', _h)
        
        ##########################
        # Network specifications #
        ##########################
        # generally, walkbacks should be 2*layers
        if self.layers % 2 == 0:
            if self.walkbacks < 2*self.layers:
                log.warning('Not enough walkbacks for the layers! Layers is %s and walkbacks is %s. '
                            'Generaly want 2X walkbacks to layers',
                            str(self.layers), str(self.walkbacks))
        else:
            if self.walkbacks < 2*self.layers-1:
                log.warning('Not enough walkbacks for the layers! Layers is %s and walkbacks is %s. '
                            'Generaly want 2X walkbacks to layers',
                            str(self.layers), str(self.walkbacks))

        self.noise_annealing        = cast32(self.noise_annealing)  # noise schedule parameter
        self.hidden_add_noise_sigma = sharedX(cast32(self.hidden_add_noise_sigma))
        self.input_salt_and_pepper  = sharedX(cast32(self.input_salt_and_pepper))

        # determine the sizes of each layer in a list.
        #  layer sizes, from h0 to hK (h0 is the visible layer)
        self.layer_sizes = [self.N_input] + [self.hidden_size] * self.layers

        #########################
        # Activation functions! #
        #########################
        # hidden unit activation
        if callable(self.hidden_activation):
            log.debug('Using specified activation for hiddens')
        elif isinstance(self.args.get('hidden_activation'), basestring):
            self.hidden_activation = get_activation_function(self.hidden_activation)
            log.debug('Using %s activation for hiddens', self.hidden_activation)
        else:
            log.critical('Missing a hidden activation function!')
            raise NotImplementedError('Missing a hidden activation function!')

        # Visible layer activation
        if callable(self.visible_activation):
            log.debug('Using specified activation for visible layer')
        elif isinstance(self.visible_activation, basestring):
            self.visible_activation = get_activation_function(self.visible_activation)
            log.debug('Using %s activation for visible layer', self.visible_activation)
        else:
            log.critical('Missing a visible activation function!')
            raise NotImplementedError('Missing a visible activation function!')

        # Cost function
        if callable(self.cost_function):
            log.debug('Using specified cost function')
        elif isinstance(self.args.get('cost_function'), basestring):
            self.cost_function = get_cost_function(self.cost_function)
            log.debug('Using %s cost function', self.cost_function)
        else:
            log.critical('Missing a cost function!')
            raise NotImplementedError('Missing a cost function!')

        ############################
        # Theano variables and RNG #
        ############################
        if self.inputs_hook is None:
            self.X = T.fmatrix('X')
        else:
            # inputs_hook is a (shape, input) tuple
            self.X = self.inputs_hook[1]

        # initialize and seed rng
        rng.seed(1)

        ###############
        # Parameters! #
        ###############
        # initialize a list of weights and biases based on layer_sizes for the GSN
        if self.weights_init.lower() == 'uniform':
            self.weights_list = [get_weights_uniform(shape=(self.layer_sizes[i], self.layer_sizes[i + 1]),
                                                     name="W_{0!s}_{1!s}".format(i, i + 1),
                                                     interval=self.weights_interval)
                                 for i in range(self.layers)]
        # if the weights should be gaussian
        elif self.weights_init.lower() == 'gaussian':
            self.weights_list = [get_weights_gaussian(shape=(self.layer_sizes[i], self.layer_sizes[i + 1]),
                                                      name="W_{0!s}_{1!s}".format(i, i + 1),
                                                      mean=self.weights_mean,
                                                      std=self.weights_std)
                                 for i in range(self.layers)]
        # otherwise not implemented
        else:
            log.error("Did not recognize weights_init %s! Pleas try gaussian or uniform" %
                      str(self.weights_init))
            raise NotImplementedError("Did not recognize weights_init %s! Pleas try gaussian or uniform" %
                                      str(self.weights_init))

        # initialize each layer bias to 0's.
        self.bias_list = [get_bias(shape=(self.layer_sizes[i],),
                                   name='b_' + str(i),
                                   init_values=self.bias_init)
                          for i in range(self.layers + 1)]

        # build the params of the model into a list
        self.params = self.weights_list + self.bias_list
        log.debug("gsn params: %s", str(self.params))

        # using the properties, build the computational graph
        self.build_computation_graph()


    def build_computation_graph(self):
        #################
        # Build the GSN #
        #################
        log.debug("Building GSN graphs...")
        # GSN for training - with noise
        add_noise = True
        # if there is no hiddens_hook,
        if self.hiddens_hook is None:
            p_X_chain, _ = GSN.build_gsn(self.X,
                                         self.weights_list,
                                         self.bias_list,
                                         add_noise,
                                         self.noiseless_h1,
                                         self.hidden_add_noise_sigma,
                                         self.input_salt_and_pepper,
                                         self.input_sampling,
                                         self.MRG,
                                         self.visible_activation,
                                         self.hidden_activation,
                                         self.walkbacks)

        # if there is a hiddens_hook, we want to change the order layers are updated and make this purely
        # generative from the hiddens
        else:
            p_X_chain, _, _, _ = GSN.build_gsn_given_hiddens(self.X,
                                                             self.unpack_hiddens(self.hiddens_hook[1]),
                                                             self.weights_list,
                                                             self.bias_list,
                                                             add_noise,
                                                             self.noiseless_h1,
                                                             self.hidden_add_noise_sigma,
                                                             self.input_salt_and_pepper,
                                                             self.input_sampling,
                                                             self.MRG,
                                                             self.visible_activation,
                                                             self.hidden_activation,
                                                             self.walkbacks,
                                                             self.cost_function)

        # GSN for prediction - no noise
        add_noise = False
        # deal with hiddens_hook exactly as above.
        if self.hiddens_hook is None:
            p_X_chain_recon, recon_hiddens = GSN.build_gsn(self.X,
                                                           self.weights_list,
                                                           self.bias_list,
                                                           add_noise,
                                                           self.noiseless_h1,
                                                           self.hidden_add_noise_sigma,
                                                           self.input_salt_and_pepper,
                                                           self.input_sampling,
                                                           self.MRG,
                                                           self.visible_activation,
                                                           self.hidden_activation,
                                                           self.walkbacks)
        else:
            p_X_chain_recon, recon_hiddens, _, _ = GSN.build_gsn_given_hiddens(self.X,
                                                                               self.unpack_hiddens(self.hiddens_hook[1]),
                                                                               self.weights_list,
                                                                               self.bias_list,
                                                                               add_noise,
                                                                               self.noiseless_h1,
                                                                               self.hidden_add_noise_sigma,
                                                                               self.input_salt_and_pepper,
                                                                               self.input_sampling,
                                                                               self.MRG,
                                                                               self.visible_activation,
                                                                               self.hidden_activation,
                                                                               self.walkbacks,
                                                                               self.cost_function)

        ####################
        # Costs and output #
        ####################
        log.debug('Cost w.r.t p(X|...) at every step in the graph for the GSN')
        # use the noisy ones for training cost
        costs          = [self.cost_function(rX, self.X) for rX in p_X_chain]
        self.show_cost = costs[-1]  # for a monitor to show progress
        self.cost      = numpy.sum(costs)  # THIS IS THE TRAINING COST - RECONSTRUCTION OF OUTPUT FROM NOISY GRAPH

        # use the non-noisy graph for prediction
        gsn_costs_recon = [self.cost_function(rX, self.X) for rX in p_X_chain_recon]
        # another monitor, same as self.show_cost but on the non-noisy graph.
        self.monitor    = gsn_costs_recon[-1]
        # this should be considered the main output of the computation, the sample after the
        # last walkback from the non-noisy graph.
        self.output     = p_X_chain_recon[-1]
        # these should be considered the model's hidden representation - the hidden representation after
        # the last walkback from the non-noisy graph.
        self.hiddens    = recon_hiddens


        self.monitors = OrderedDict([('noisy_recon_cost', self.show_cost), ('recon_cost', self.monitor)])
        

        ############
        # Sampling #
        ############
        # the input to the sampling function
        X_sample = T.fmatrix("X_sampling")
        self.network_state_input = [X_sample] + [T.fmatrix("H_sampling_"+str(i+1)) for i in range(self.layers)]
       
        # "Output" state of the network (noisy)
        # initialized with input, then we apply updates
        self.network_state_output = [X_sample] + self.network_state_input[1:]
        visible_pX_chain = []
    
        # ONE update
        log.debug("Performing one walkback in network state sampling.")
        GSN.update_layers(self.network_state_output,
                          self.weights_list,
                          self.bias_list,
                          visible_pX_chain,
                          True,
                          self.noiseless_h1,
                          self.hidden_add_noise_sigma,
                          self.input_salt_and_pepper,
                          self.input_sampling,
                          self.MRG,
                          self.visible_activation,
                          self.hidden_activation)

        #####################################################
        #     Create the predict and monitor functions      #
        #####################################################
        log.debug("Compiling functions...")
        t = time.time()

        # doesn't make sense to have this if there is a hiddens_hook
        if self.hiddens_hook is None:
            # THIS IS THE MAIN PREDICT FUNCTION - takes in a real matrix and produces the output from the non-noisy
            # computation graph
            log.debug("f_predict...")
            self.f_predict = function(inputs  = [self.X],
                                      outputs = self.output,
                                      name    = 'gsn_f_predict')

        # this is a helper function - it corrupts inputs when testing the non-noisy graph (aka before feeding the
        # input to f_predict)
        log.debug("f_noise...")
        self.f_noise = function(inputs  = [self.X],
                                outputs = salt_and_pepper(self.X, self.input_salt_and_pepper, self.MRG),
                                name    = 'gsn_f_noise')

        # the sampling function, for creating lots of samples from the computational graph. (mostly for log-likelihood
        # or visualization)
        log.debug("f_sample...")
        if self.layers == 1: 
            self.f_sample = function(inputs  = [X_sample],
                                     outputs = visible_pX_chain[-1],
                                     name    = 'gsn_f_sample_single_layer')
        else:
            # WHY IS THERE A WARNING????
            # because the first odd layers are not used -> directly computed FROM THE EVEN layers
            # unused input = warn
            self.f_sample = function(inputs  = self.network_state_input,
                                     outputs = self.network_state_output + visible_pX_chain,
                                     name    = 'gsn_f_sample')

        log.debug("GSN compiling done. Took %s", make_time_units_string(time.time() - t))

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
        return [self.X]

    def get_hiddens(self):
        """
        This method will return the model's hidden representation expression (if applicable) from the
        computational graph.

        This will also be used for creating hooks to link models together, where these hidden variables can be strung
        as the inputs or hiddens to another model :)
        ------------------

        :return: theano expression of the hidden representation from this model's computation
        :rtype: theano tensor (expression)
        """
        if not hasattr(self, 'hiddens'):
            log.error("Missing self.hiddens - make sure you ran self.build_computation_graph()! "
                      "This should have run during initialization....")
            raise NotImplementedError()
        return self.pack_hiddens(self.hiddens)

    def get_outputs(self):
        """
        This method will return the model's output variable expression from the computational graph. This should be
        what is given for the outputs= part of the 'f_predict' function from self.predict().

        This will be used for creating hooks to link models together, where these outputs can be strung as the inputs
        or hiddens to another model :)
        ------------------

        :return: theano expression of the outputs from this model's computation
        :rtype: theano tensor (expression)
        """
        if not hasattr(self, 'output'):
            log.error(
                "Missing self.output - make sure you ran self.build_computation_graph()! "
                "This should have run during initialization....")
            raise NotImplementedError()
        return self.output

    def get_train_cost(self):
        """
        This returns the expression that represents the cost given an input, which is used for the Optimizer during
        training. The reason we can't just compile a f_train theano function is because updates need to be calculated
        for the parameters during gradient descent - and these updates are created in the Optimizer object.
        ------------------

        :return: theano expression of the model's training cost, from which parameter gradients will be computed.
        :rtype: theano tensor
        """
        return self.cost

    def get_monitors(self):
        """
        This returns a dictionary of (monitor_name: monitor_expression) of variables (monitors) whose values we care
        about during training. For every monitor returned by this method, the function will be run on the
        train/validation/test dataset and its value will be reported.
        ------------------

        :return: Dictionary of String: theano_function for each monitor variable we care about in the model.
        :rtype: Dictionary
        """
        return self.monitors

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
        # noise scheduling
        noise_schedule = get_decay_function(self.noise_decay,
                                            self.input_salt_and_pepper,
                                            self.args.get('input_salt_and_pepper'),
                                            self.noise_annealing)
        return [noise_schedule]

    def get_params(self):
        """
        This returns the list of theano shared variables that will be trained by the Optimizer.
        These parameters are used in the gradient.
        ------------------

        :return: flattened list of theano shared variables to be trained
        :rtype: List(shared_variables)
        """
        return self.params

    
    # def gen_10k_samples(self):
    #     log.info('Generating 10,000 samples')
    #     samples, _ = self.sample(self.test_X[0].get_value()[1:2], 10000, 1)
    #     f_samples = 'samples.npy'
    #     numpy.save(f_samples, samples)
    #     log.debug('saved digits')
    #
    # def sample(self, initial, n_samples=400, k=1):
    #     log.debug("Starting sampling...")
    #     def sample_some_numbers_single_layer(n_samples):
    #         x0 = initial
    #         samples = [x0]
    #         x = self.f_noise(x0)
    #         for _ in xrange(n_samples-1):
    #             x = self.f_sample(x)
    #             samples.append(x)
    #             x = rng.binomial(n=1, p=x, size=x.shape).astype('float32')
    #             x = self.f_noise(x)
    #
    #         log.debug("Sampling done.")
    #         return numpy.vstack(samples), None
    #
    #     def sampling_wrapper(NSI):
    #         # * is the "splat" operator: It takes a list as input, and expands it into actual
    #         # positional arguments in the function call.
    #         out = self.f_sample(*NSI)
    #         NSO = out[:len(self.network_state_output)]
    #         vis_pX_chain = out[len(self.network_state_output):]
    #         return NSO, vis_pX_chain
    #
    #     def sample_some_numbers(n_samples):
    #         # The network's initial state
    #         init_vis       = initial
    #         noisy_init_vis = self.f_noise(init_vis)
    #
    #         network_state  = [[noisy_init_vis] + [numpy.zeros((initial.shape[0],self.hidden_size), dtype='float32')
    #                           for _ in self.bias_list[1:]]]
    #
    #         visible_chain  = [init_vis]
    #         noisy_h0_chain = [noisy_init_vis]
    #         sampled_h = []
    #
    #         times = []
    #         for i in xrange(n_samples-1):
    #             _t = time.time()
    #
    #             # feed the last state into the network, compute new state, and obtain visible units expectation chain
    #             net_state_out, vis_pX_chain = sampling_wrapper(network_state[-1])
    #
    #             # append to the visible chain
    #             visible_chain += vis_pX_chain
    #
    #             # append state output to the network state chain
    #             network_state.append(net_state_out)
    #
    #             noisy_h0_chain.append(net_state_out[0])
    #
    #             if i%k == 0:
    #                 sampled_h.append(T.stack(net_state_out[1:]))
    #                 if i == k:
    #                     log.debug("About "+make_time_units_string(numpy.mean(times)*(n_samples-1-i))+" remaining...")
    #
    #             times.append(time.time() - _t)
    #
    #         log.DEBUG("Sampling done.")
    #         return numpy.vstack(visible_chain), sampled_h
    #
    #     if self.layers == 1:
    #         return sample_some_numbers_single_layer(n_samples)
    #     else:
    #         return sample_some_numbers(n_samples)
    #
    # def plot_samples(self, epoch_number="", leading_text="", n_samples=400):
    #     to_sample = time.time()
    #     initial = self.test_X[0].get_value(borrow=True)[:1]
    #     rand_idx = numpy.random.choice(range(self.test_X[0].get_value(borrow=True).shape[0]))
    #     rand_init = self.test_X[0].get_value(borrow=True)[rand_idx:rand_idx+1]
    #
    #     V, _ = self.sample(initial, n_samples)
    #     rand_V, _ = self.sample(rand_init, n_samples)
    #
    #     img_samples = PIL.Image.fromarray(
    #       tile_raster_images(V, (self.image_height, self.image_width), closest_to_square_factors(n_samples))
    #     )
    #     rand_img_samples = PIL.Image.fromarray(
    #       tile_raster_images(rand_V, (self.image_height, self.image_width), closest_to_square_factors(n_samples))
    #     )
    #
    #     fname = self.outdir+leading_text+'samples_epoch_'+str(epoch_number)+'.png'
    #     img_samples.save(fname)
    #     rfname = self.outdir+leading_text+'samples_rand_epoch_'+str(epoch_number)+'.png'
    #     rand_img_samples.save(rfname)
    #     log.debug('Took ' + make_time_units_string(time.time() - to_sample) +
    #               ' to sample '+str(n_samples*2)+' numbers')

    def create_reconstruction_image(self, input_data):
        if self.is_image:
            n_examples = len(input_data)
            xs_test = input_data
            noisy_xs_test = self.f_noise(input_data)
            reconstructed = self.predict(noisy_xs_test)
            # Concatenate stuff
            width, height = closest_to_square_factors(n_examples)
            stacked = numpy.vstack(
                [numpy.vstack([xs_test[i * width: (i + 1) * width],
                               noisy_xs_test[i * width: (i + 1) * width],
                               reconstructed[i * width: (i + 1) * width]])
                 for i in range(height)])
            number_reconstruction = PIL.Image.fromarray(
                tile_raster_images(stacked, (self.image_height, self.image_width), (height, 3*width))
            )

            save_path = os.path.join(self.outdir, 'reconstruction.png')
            save_path = os.path.realpath(save_path)
            number_reconstruction.save(save_path)
            log.info("saved output image to %s", save_path)
        else:
            log.warning("Asked to create reconstruction image for something that isn't an image. Ignoring...")


    def pack_hiddens(self, hiddens_list):
        '''
        This concatenates all the odd layers into a single tensor
        (GSNs alternate even/odd layers for storing network state)

        :param hiddens_list: list of the hiddens [h0...hn] where h0 is the visible layer
        :type hiddens_list: List(theano tensor)

        :return: tensor concatenating the appropriate layers
        :rtype: theano tensor
        '''
        output_list = []
        for idx, layer in enumerate(hiddens_list):
            # we care about the odd hidden layers (since the visible layer is h0)
            if idx % 2 != 0:
                output_list.append(layer)

        hiddens_tensor = T.concatenate(output_list, axis=1)
        return hiddens_tensor


    def unpack_hiddens(self, hiddens_tensor):
        '''
        This makes a tensor of the hidden layers into a list

        :param hiddens_tensor: theano tensor containing the odd layers of the gsn concatenated
        :type hiddens_tensor: theano tensor

        :return: list of theano variables that make the hidden representation
        (including the even layers initialized to 0)
        :rtype: List(theano tensor)
        '''
        h_list = [T.zeros_like(self.X)]
        for idx, w in enumerate(self.weights_list):
            # we only care about the odd layers
            # (where h0 is the input layer - which makes it even here in the hidden layer space)
            if (idx % 2) != 0:
                h_list.append(T.zeros_like(T.dot(h_list[-1], w)))
            else:
                h_list.append((hiddens_tensor.T[(idx/2)*self.layer_sizes[idx] : (idx/2+1)*self.layer_sizes[idx+1]]).T)

        return h_list


    def save_params(self, param_file):
        """
        This saves the model's parameters to the param_file (pickle file)
        ------------------

        :param param_file: filename of pickled params file
        :type param_file: String

        :return: whether or not successful
        :rtype: Boolean
        """
        # save to the output directory from the model config
        base = self.outdir
        filepath = os.path.join(base, param_file)
        super(GSN, self).save_params(filepath)


    def save_args(self, args_file="gsn_config.pkl"):
        # save to the output directory from the model config
        base = self.outdir
        filepath = os.path.join(base, args_file)
        super(GSN, self).save_args(filepath)


###############################################
# COMPUTATIONAL GRAPH HELPER METHODS FOR GSN #
###############################################
    @staticmethod
    def update_layers(hiddens,
                      weights_list,
                      bias_list,
                      p_X_chain,
                      add_noise              = _defaults["add_noise"],
                      noiseless_h1           = _defaults["noiseless_h1"],
                      hidden_add_noise_sigma = _defaults["hidden_add_noise_sigma"],
                      input_salt_and_pepper  = _defaults["input_salt_and_pepper"],
                      input_sampling         = _defaults["input_sampling"],
                      MRG                    = _defaults["MRG"],
                      visible_activation     = _defaults["visible_activation"],
                      hidden_activation      = _defaults["hidden_activation"]):
        # One update over the odd layers + one update over the even layers
        log.debug('odd layer updates')
        # update the odd layers
        GSN.update_odd_layers(hiddens,
                              weights_list,
                              bias_list,
                              add_noise,
                              noiseless_h1,
                              hidden_add_noise_sigma,
                              input_salt_and_pepper,
                              input_sampling,
                              MRG,
                              visible_activation,
                              hidden_activation)
        log.debug('even layer updates')
        # update the even layers
        GSN.update_even_layers(hiddens,
                               weights_list,
                               bias_list,
                               p_X_chain,
                               add_noise,
                               noiseless_h1,
                               hidden_add_noise_sigma,
                               input_salt_and_pepper,
                               input_sampling,
                               MRG,
                               visible_activation,
                               hidden_activation)
        log.debug('done full update.')

    @staticmethod
    def update_layers_scan_step(hiddens_t,
                                weights_list,
                                bias_list,
                                add_noise              = _defaults["add_noise"],
                                noiseless_h1           = _defaults["noiseless_h1"],
                                hidden_add_noise_sigma = _defaults["hidden_add_noise_sigma"],
                                input_salt_and_pepper  = _defaults["input_salt_and_pepper"],
                                input_sampling         = _defaults["input_sampling"],
                                MRG                    = _defaults["MRG"],
                                visible_activation     = _defaults["visible_activation"],
                                hidden_activation      = _defaults["hidden_activation"]):
        p_X_chain = []
        log.debug("One full update step for layers.")
        # One update over the odd layers + one update over the even layers
        log.debug('odd layer updates')
        # update the odd layers
        GSN.update_odd_layers(hiddens_t,
                              weights_list,
                              bias_list,
                              add_noise,
                              noiseless_h1,
                              hidden_add_noise_sigma,
                              input_salt_and_pepper,
                              input_sampling,
                              MRG,
                              visible_activation,
                              hidden_activation)
        log.debug('even layer updates')
        # update the even layers
        GSN.update_even_layers(hiddens_t,
                               weights_list,
                               bias_list,
                               p_X_chain,
                               add_noise,
                               noiseless_h1,
                               hidden_add_noise_sigma,
                               input_salt_and_pepper,
                               input_sampling,
                               MRG,
                               visible_activation,
                               hidden_activation)
        log.debug('done full update.')
        # return the generated sample, the sampled next input, and hiddens
        return p_X_chain[0], hiddens_t

    @staticmethod
    def update_layers_reverse(hiddens,
                              weights_list,
                              bias_list,
                              p_X_chain,
                              add_noise              = _defaults["add_noise"],
                              noiseless_h1           = _defaults["noiseless_h1"],
                              hidden_add_noise_sigma = _defaults["hidden_add_noise_sigma"],
                              input_salt_and_pepper  = _defaults["input_salt_and_pepper"],
                              input_sampling         = _defaults["input_sampling"],
                              MRG                    = _defaults["MRG"],
                              visible_activation     = _defaults["visible_activation"],
                              hidden_activation      = _defaults["hidden_activation"]):
        # One update over the even layers + one update over the odd layers
        log.debug('even layer updates')
        # update the even layers
        GSN.update_even_layers(hiddens,
                               weights_list,
                               bias_list,
                               p_X_chain,
                               add_noise,
                               noiseless_h1,
                               hidden_add_noise_sigma,
                               input_salt_and_pepper,
                               input_sampling,
                               MRG,
                               visible_activation,
                               hidden_activation)
        log.debug('odd layer updates')
        # update the odd layers
        GSN.update_odd_layers(hiddens,
                              weights_list,
                              bias_list,
                              add_noise,
                              noiseless_h1,
                              hidden_add_noise_sigma,
                              input_salt_and_pepper,
                              input_sampling,
                              MRG,
                              visible_activation,
                              hidden_activation)
        log.debug('done full update.')


    # Odd layer update function
    # just a loop over the odd layers
    @staticmethod
    def update_odd_layers(hiddens,
                          weights_list,
                          bias_list,
                          add_noise              = _defaults["add_noise"],
                          noiseless_h1           = _defaults["noiseless_h1"],
                          hidden_add_noise_sigma = _defaults["hidden_add_noise_sigma"],
                          input_salt_and_pepper  = _defaults["input_salt_and_pepper"],
                          input_sampling         = _defaults["input_sampling"],
                          MRG                    = _defaults["MRG"],
                          visible_activation     = _defaults["visible_activation"],
                          hidden_activation      = _defaults["hidden_activation"]):
        # Loop over the odd layers
        for i in range(1, len(hiddens), 2):
            log.debug('updating layer %s', str(i))
            GSN.simple_update_layer(hiddens,
                                    weights_list,
                                    bias_list,
                                    None,
                                    i,
                                    add_noise,
                                    noiseless_h1,
                                    hidden_add_noise_sigma,
                                    input_salt_and_pepper,
                                    input_sampling,
                                    MRG,
                                    visible_activation,
                                    hidden_activation)

    # Even layer update
    # p_X_chain is given to append the p(X|...) at each full update (one update = odd update + even update)
    @staticmethod
    def update_even_layers(hiddens,
                           weights_list,
                           bias_list,
                           p_X_chain,
                           add_noise              = _defaults["add_noise"],
                           noiseless_h1           = _defaults["noiseless_h1"],
                           hidden_add_noise_sigma = _defaults["hidden_add_noise_sigma"],
                           input_salt_and_pepper  = _defaults["input_salt_and_pepper"],
                           input_sampling         = _defaults["input_sampling"],
                           MRG                    = _defaults["MRG"],
                           visible_activation     = _defaults["visible_activation"],
                           hidden_activation      = _defaults["hidden_activation"]):
        # Loop over even layers
        for i in range(0, len(hiddens), 2):
            log.debug('updating layer %s', str(i))
            GSN.simple_update_layer(hiddens,
                                    weights_list,
                                    bias_list,
                                    p_X_chain,
                                    i,
                                    add_noise,
                                    noiseless_h1,
                                    hidden_add_noise_sigma,
                                    input_salt_and_pepper,
                                    input_sampling,
                                    MRG,
                                    visible_activation,
                                    hidden_activation)


    # The layer update function
    # hiddens   :   list containing the symbolic theano variables [visible, hidden1, hidden2, ...]
    #               layer_update will modify this list inplace
    # weights_list : list containing the theano variables weights between hidden layers
    # bias_list :   list containing the theano variables bias corresponding to hidden layers
    # p_X_chain :   list containing the successive p(X|...) at each update
    #               update_layer will append to this list
    # i         :   the current layer being updated
    # add_noise :   pre (and post) activation gaussian noise flag
    @staticmethod
    def simple_update_layer(hiddens,
                            weights_list,
                            bias_list,
                            p_X_chain,
                            i,
                            add_noise              = _defaults["add_noise"],
                            noiseless_h1           = _defaults["noiseless_h1"],
                            hidden_add_noise_sigma = _defaults["hidden_add_noise_sigma"],
                            input_salt_and_pepper  = _defaults["input_salt_and_pepper"],
                            input_sampling         = _defaults["input_sampling"],
                            MRG                    = _defaults["MRG"],
                            visible_activation     = _defaults["visible_activation"],
                            hidden_activation      = _defaults["hidden_activation"]):
        # Compute the dot product, whatever layer
        # If the visible layer X
        if i == 0:
            log.debug('using '+str(weights_list[i])+'.T')
            hiddens[i] = T.dot(hiddens[i+1], weights_list[i].T) + bias_list[i]
        # If the top layer
        elif i == len(hiddens)-1:
            log.debug('using '+str(weights_list[i-1]))
            hiddens[i] = T.dot(hiddens[i-1], weights_list[i-1]) + bias_list[i]
        # Otherwise in-between layers
        else:
            log.debug("using %s and %s.T", str(weights_list[i-1]), str(weights_list[i]))
            # next layer        :   hiddens[i+1], assigned weights : W_i
            # previous layer    :   hiddens[i-1], assigned weights : W_(i-1)
            hiddens[i] = T.dot(hiddens[i+1], weights_list[i].T) + T.dot(hiddens[i-1], weights_list[i-1]) + bias_list[i]

        # Add pre-activation noise if NOT input layer
        if i == 1 and noiseless_h1:
            log.debug('>>NO noise in first hidden layer')
            add_noise = False

        # pre activation noise
        if i != 0 and add_noise:
            log.debug('Adding pre-activation gaussian noise for layer %s', str(i))
            hiddens[i] = add_gaussian(hiddens[i], std=hidden_add_noise_sigma, MRG=MRG)

        # ACTIVATION!
        if i == 0:
            log.debug('Activation for visible layer')
            hiddens[i] = visible_activation(hiddens[i])
        else:
            log.debug('Hidden units activation for layer %s', str(i))
            hiddens[i] = hidden_activation(hiddens[i])

        # post activation noise
        # why is there post activation noise? Because there is already pre-activation noise,
        # this just doubles the amount of noise between each activation of the hiddens.
        if i != 0 and add_noise:
            log.debug('Adding post-activation gaussian noise for layer %s', str(i))
            hiddens[i] = add_gaussian(hiddens[i], std=hidden_add_noise_sigma, MRG=MRG)

        # build the reconstruction chain if updating the visible layer X
        if i == 0:
            # if input layer -> append p(X|H...)
            p_X_chain.append(hiddens[i])

            # sample from p(X|H...) - SAMPLING NEEDS TO BE CORRECT FOR INPUT TYPES
            # I.E. FOR BINARY MNIST SAMPLING IS BINOMIAL. real-valued inputs should be gaussian
            if input_sampling:
                log.debug('Sampling from input')
                sampled = MRG.binomial(p=hiddens[i], size=hiddens[i].shape, dtype='float32')
            else:
                log.debug('>>NO input sampling')
                sampled = hiddens[i]
            # add noise
            sampled = salt_and_pepper(sampled, input_salt_and_pepper, MRG)

            # set input layer
            hiddens[i] = sampled



    ############################
    #   THE MAIN GSN BUILDER   #
    ############################
    @staticmethod
    def build_gsn(X,
                  weights_list,
                  bias_list,
                  add_noise              = _defaults["add_noise"],
                  noiseless_h1           = _defaults["noiseless_h1"],
                  hidden_add_noise_sigma = _defaults["hidden_add_noise_sigma"],
                  input_salt_and_pepper  = _defaults["input_salt_and_pepper"],
                  input_sampling         = _defaults["input_sampling"],
                  MRG                    = _defaults["MRG"],
                  visible_activation     = _defaults["visible_activation"],
                  hidden_activation      = _defaults["hidden_activation"],
                  walkbacks              = _defaults["walkbacks"]):
        """
        Construct a GSN (unimodal transition operator) for k walkbacks on the input X.
        Returns the list of predicted X's after k walkbacks and the resulting layer values.

        @type  X: Theano symbolic variable
        @param X: The variable representing the visible input.

        @type  weights_list: List(matrix)
        @param weights_list: The list of weights to use between layers.

        @type  bias_list: List(vector)
        @param bias_list: The list of biases to use for each layer.

        @type  add_noise: Boolean
        @param add_noise: Whether or not to add noise in the computational graph.

        @type  noiseless_h1: Boolean
        @param noiseless_h1: Whether or not to add noise in the first hidden layer.

        @type  hidden_add_noise_sigma: Float
        @param hidden_add_noise_sigma: The sigma value for the hidden noise function.

        @type  input_salt_and_pepper: Float
        @param input_salt_and_pepper: The amount of masking noise to use.

        @type  input_sampling: Boolean
        @param input_sampling: Whether to sample from each walkback prediction (like Gibbs).

        @type  MRG: Theano random generator
        @param MRG: Random generator.

        @type  visible_activation: Function
        @param visible_activation: The visible layer X activation function.

        @type  hidden_activation: Function
        @param hidden_activation: The hidden layer activation function.

        @type  walkbacks: Integer
        @param walkbacks: The k number of walkbacks to use for the GSN.

        @rtype:   List
        @return:  predicted_x_chain, hiddens
        """
        p_X_chain = []
        # Whether or not to corrupt the visible input X
        if add_noise:
            X_init = salt_and_pepper(X, input_salt_and_pepper, MRG)
        else:
            X_init = X
        # init hiddens with zeros
        hiddens = [X_init]
        for w in weights_list:
            hiddens.append(T.zeros_like(T.dot(hiddens[-1], w)))
        # The layer update scheme
        log.info("Building the GSN graph : %s updates", str(walkbacks))
        for i in range(walkbacks):
            log.debug("GSN Walkback %s/%s", str(i+1), str(walkbacks))
            GSN.update_layers(hiddens,
                              weights_list,
                              bias_list,
                              p_X_chain,
                              add_noise,
                              noiseless_h1,
                              hidden_add_noise_sigma,
                              input_salt_and_pepper,
                              input_sampling,
                              MRG,
                              visible_activation,
                              hidden_activation)

        return p_X_chain, hiddens

    @staticmethod
    def build_gsn_given_hiddens(X,
                                hiddens,
                                weights_list,
                                bias_list,
                                add_noise              = _defaults["add_noise"],
                                noiseless_h1           = _defaults["noiseless_h1"],
                                hidden_add_noise_sigma = _defaults["hidden_add_noise_sigma"],
                                input_salt_and_pepper  = _defaults["input_salt_and_pepper"],
                                input_sampling         = _defaults["input_sampling"],
                                MRG                    = _defaults["MRG"],
                                visible_activation     = _defaults["visible_activation"],
                                hidden_activation      = _defaults["hidden_activation"],
                                walkbacks              = _defaults["walkbacks"],
                                cost_function          = _defaults["cost_function"]):

        log.info("Building the GSN graph given hiddens with %s walkbacks", str(walkbacks))
        p_X_chain = []
        for i in range(walkbacks):
            log.debug("GSN (prediction) Walkback %s/%s", str(i+1), str(walkbacks))
            GSN.update_layers_reverse(hiddens,
                                      weights_list,
                                      bias_list,
                                      p_X_chain,
                                      add_noise,
                                      noiseless_h1,
                                      hidden_add_noise_sigma,
                                      input_salt_and_pepper,
                                      input_sampling,
                                      MRG,
                                      visible_activation,
                                      hidden_activation)

        # x_sample = p_X_chain[-1]

        costs     = [cost_function(rX, X) for rX in p_X_chain]
        show_cost = costs[-1] # for logging to show progress
        cost      = numpy.sum(costs)

        return p_X_chain, hiddens, cost, show_cost

    @staticmethod
    def build_gsn_scan(X,
                       weights_list,
                       bias_list,
                       add_noise              = _defaults["add_noise"],
                       noiseless_h1           = _defaults["noiseless_h1"],
                       hidden_add_noise_sigma = _defaults["hidden_add_noise_sigma"],
                       input_salt_and_pepper  = _defaults["input_salt_and_pepper"],
                       input_sampling         = _defaults["input_sampling"],
                       MRG                    = _defaults["MRG"],
                       visible_activation     = _defaults["visible_activation"],
                       hidden_activation      = _defaults["hidden_activation"],
                       walkbacks              = _defaults["walkbacks"],
                       cost_function          = _defaults["cost_function"]):

        # Whether or not to corrupt the visible input X
        if add_noise:
            X_init = salt_and_pepper(X, input_salt_and_pepper, MRG)
        else:
            X_init = X
        # init hiddens with zeros
        hiddens_0 = [X_init]
        for w in weights_list:
            hiddens_0.append(T.zeros_like(T.dot(hiddens_0[-1], w)))

        log.info("Building the GSN graph (for scan) with %s walkbacks", str(walkbacks))
        p_X_chain = []
        for i in range(walkbacks):
            log.debug("GSN (after scan) Walkback %s/%s", str(i+1), str(walkbacks))
            GSN.update_layers(hiddens_0,
                              weights_list,
                              bias_list,
                              p_X_chain,
                              add_noise,
                              noiseless_h1,
                              hidden_add_noise_sigma,
                              input_salt_and_pepper,
                              input_sampling,
                              MRG,
                              visible_activation,
                              hidden_activation)


        x_sample = p_X_chain[-1]

        costs     = [cost_function(rX, X) for rX in p_X_chain]
        show_cost = costs[-1] # for logging to show progress
        cost      = numpy.sum(costs)

        return x_sample, cost, show_cost#, updates

    @staticmethod
    def build_gsn_pxh(hiddens,
                    weights_list,
                    bias_list,
                    add_noise              = _defaults["add_noise"],
                    noiseless_h1           = _defaults["noiseless_h1"],
                    hidden_add_noise_sigma = _defaults["hidden_add_noise_sigma"],
                    input_salt_and_pepper  = _defaults["input_salt_and_pepper"],
                    input_sampling         = _defaults["input_sampling"],
                    MRG                    = _defaults["MRG"],
                    visible_activation     = _defaults["visible_activation"],
                    hidden_activation      = _defaults["hidden_activation"],
                    walkbacks              = _defaults["walkbacks"]):

        log.info("Building the GSN graph for P(X=x|H) with %s walkbacks", str(walkbacks))
        p_X_chain = []
        for i in range(walkbacks):
            log.debug("GSN Walkback %s/%s", str(i+1), str(walkbacks))
            GSN.update_layers(hiddens,
                              weights_list,
                              bias_list,
                              p_X_chain,
                              add_noise,
                              noiseless_h1,
                              hidden_add_noise_sigma,
                              input_salt_and_pepper,
                              input_sampling,
                              MRG,
                              visible_activation,
                              hidden_activation)

        x_sample = p_X_chain[-1]

        return x_sample