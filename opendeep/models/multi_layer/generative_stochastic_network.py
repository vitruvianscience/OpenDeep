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
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG
from theano.compat.python2x import OrderedDict
import PIL
# internal references
from opendeep import cast_floatX, function, sharedX
from opendeep.models.model import Model
from opendeep.utils.decay import get_decay_function
from opendeep.utils.activation import get_activation_function, is_binary
from opendeep.utils.cost import get_cost_function
from opendeep.utils.misc import closest_to_square_factors, make_time_units_string, raise_to_list
from opendeep.utils.nnet import get_weights, get_bias
from opendeep.utils.noise import salt_and_pepper, add_gaussian
from opendeep.utils.image import tile_raster_images

log = logging.getLogger(__name__)

# Default values to use for some GSN parameters.
# These defaults are used to produce the MNIST results given in the comments top of file.
_defaults = {# gsn parameters
            "layers": 3,  # number of hidden layers to use
            "walkbacks": 5,  # number of walkbacks (generally 2*layers) - need enough to propagate to visible layer
            "input_size": None,  # number of input features - please specify for your dataset!
            "hidden_size": 1500,  # number of hidden units in each layer
            "visible_activation": 'sigmoid',  # activation for visible layer - make appropriate for input data type.
            "hidden_activation": 'tanh',  # activation for hidden layers
            "input_sampling": True,  # whether to sample at each walkback step - makes it like Gibbs sampling.
            "mrg": RNG_MRG.MRG_RandomStreams(1),  # default random number generator from Theano
            "tied_weights": True,  # whether to tie the weights between layers (use transpose from higher to lower)
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
            "outdir": 'outputs/gsn/',  # base directory to output various files
            "image_width": None,  # if the input is an image, its width
            "image_height": None,  # if the input is an image, its height
            "vis_init": False}

class GSN(Model):
    '''
    Class for creating a new Generative Stochastic Network (GSN)
    '''
    def __init__(self, config=None, defaults=_defaults,
                 inputs_hook=None, hiddens_hook=None, params_hook=None,
                 input_size=None, hidden_size=None,
                 layers=None, walkbacks=None,
                 visible_activation=None,
                 hidden_activation=None,
                 input_sampling=None, mrg=None,
                 tied_weights=None, weights_init=None, weights_interval=None, weights_mean=None, weights_std=None,
                 bias_init=None,
                 cost_function=None,
                 noise_decay=None, noise_annealing=None,
                 add_noise=None, noiseless_h1=None, hidden_add_noise_sigma=None, input_salt_and_pepper=None,
                 outdir=None,
                 image_width=None, image_height=None,
                 vis_init=None):
        # init Model to combine the defaults and config dictionaries with the initial parameters.
        initial_parameters = locals().copy()
        initial_parameters.pop('self')
        super(GSN, self).__init__(**initial_parameters)
        # configs can now be accessed through self! huzzah!

        # variables from the dataset that are used for initialization and image reconstruction
        if self.inputs_hook is None:
            # if no inputs_hook given, look for the dimensionality of the input from the 'input_size' parameter.
            if self.input_size is None:
                log.critical("Please specify input_size.")
                raise AssertionError("Please specify input_size.")
        else:
            assert len(self.inputs_hook) == 2, "inputs_hook was expected (shape, input) tuple. " \
                                               "Found length %s instead" % str(len(self.inputs_hook))
            # otherwise grab shape from inputs_hook that was given
            self.input_size = self.inputs_hook[0]

        # when the input should be thought of as an image, either use the specified width and height,
        # or try to make as square as possible.
        if self.image_height is None and self.image_width is None:
            (_h, _w) = closest_to_square_factors(self.input_size)
            self.image_width  = _w
            self.image_height = _h

        ############################
        # Theano variables and RNG #
        ############################
        if self.inputs_hook is None:
            self.X = T.matrix('X')
        else:
            # inputs_hook is a (shape, input) tuple
            self.X = self.inputs_hook[1]
        
        ##########################
        # Network specifications #
        ##########################
        # generally, walkbacks should be at least 2*layers
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

        self.noise_annealing        = cast_floatX(self.noise_annealing)  # noise schedule parameter
        self.hidden_add_noise_sigma = sharedX(cast_floatX(self.hidden_add_noise_sigma))
        self.input_salt_and_pepper  = sharedX(cast_floatX(self.input_salt_and_pepper))

        # if there was a hiddens_hook, unpack the hidden layers in the tensor
        if self.hiddens_hook is not None:
            self.hidden_size = self.hiddens_hook[0]
            self.hiddens_flag = True
        else:
            self.hiddens_flag = False

        # determine the sizes of each layer in a list.
        #  layer sizes, from h0 to hK (h0 is the visible layer)
        hidden_size = list(raise_to_list(self.hidden_size))
        if len(hidden_size) == 1:
            self.layer_sizes = [self.input_size] + [self.hidden_size] * self.layers
        else:
            assert len(hidden_size) == self.layers, "Hiddens sizes and number of hidden layers mismatch." + \
                                                    "Hiddens %d and layers %d" % (len(hidden_size), self.layers)
            self.layer_sizes = [self.input_size] + list(self.hidden_size)

        if self.hiddens_hook is not None:
            self.hiddens = self.unpack_hiddens(self.hiddens_hook[1])

        #########################
        # Activation functions! #
        #########################
        # hidden unit activation
        self.hidden_activation = get_activation_function(self.hidden_activation)
        # Visible layer activation
        self.visible_activation = get_activation_function(self.visible_activation)
        # make sure the sampling functions are appropriate for the activation functions.
        if is_binary(self.visible_activation):
            self.visible_sampling = self.mrg.binomial
        else:
            # TODO: implement non-binary activation
            log.error("Non-binary visible activation not supported yet!")
            raise NotImplementedError("Non-binary visible activation not supported yet!")

        # Cost function
        self.cost_function = get_cost_function(self.cost_function)

        ###############
        # Parameters! #
        ###############
        # make sure to deal with params_hook!
        if self.params_hook is not None:
            # if tied weights, expect layers*2 + 1 params
            if self.tied_weights:
                assert len(self.params_hook) == 2*self.layers + 1, \
                    "Tied weights: expected {0!s} params, found {1!s}!".format(2*self.layers+1, len(self.params_hook))
                self.weights_list = self.params_hook[:self.layers]
                self.bias_list = self.params_hook[self.layers:]
            # if untied weights, expect layers*3 + 1 params
            else:
                assert len(self.params_hook) == 3*self.layers + 1, \
                    "Untied weights: expected {0!s} params, found {1!s}!".format(3*self.layers+1, len(self.params_hook))
                self.weights_list = self.params_hook[:2*self.layers]
                self.bias_list = self.params_hook[2*self.layers:]
        # otherwise, construct our params
        else:
            # initialize a list of weights and biases based on layer_sizes for the GSN
            self.weights_list = [get_weights(weights_init=self.weights_init,
                                             shape=(self.layer_sizes[i], self.layer_sizes[i+1]),
                                             name="W_{0!s}_{1!s}".format(i, i+1),
                                             # if gaussian
                                             mean=self.weights_mean,
                                             std=self.weights_std,
                                             # if uniform
                                             interval=self.weights_interval)
                                 for i in range(self.layers)]
            # add more weights if we aren't tying weights between layers (need to add for higher-lower layers now)
            if not self.tied_weights:
                self.weights_list.extend(
                    [get_weights(weights_init=self.weights_init,
                                 shape=(self.layer_sizes[i+1], self.layer_sizes[i]),
                                 name="W_{0!s}_{1!s}".format(i+1, i),
                                 # if gaussian
                                 mean=self.weights_mean,
                                 std=self.weights_std,
                                 # if uniform
                                 interval=self.weights_interval)
                     for i in reversed(range(self.layers))]
                )
            # initialize each layer bias to 0's.
            self.bias_list = [get_bias(shape=(self.layer_sizes[i],),
                                       name='b_' + str(i),
                                       init_values=self.bias_init)
                              for i in range(self.layers+1)]

        # build the params of the model into a list
        self.params = self.weights_list + self.bias_list
        log.debug("gsn params: %s", str(self.params))

        # using the properties, build the computational graph
        self.cost, self.monitors, self.output, self.hiddens = self.build_computation_graph()

    def build_computation_graph(self):
        #################
        # Build the GSN #
        #################
        log.debug("Building GSN graphs...")

        # GSN for training - with noise
        add_noise = self.add_noise
        # if there is no hiddens_hook, build the GSN normally using the input X
        if not self.hiddens_flag:
            p_X_chain, _ = self.build_gsn(add_noise=add_noise)

        # if there is a hiddens_hook, we want to change the order layers are updated and make this purely
        # generative from the hiddens
        else:
            p_X_chain, _,  = self.build_gsn(hiddens=self.hiddens, add_noise=add_noise, reverse=True)

        # GSN for prediction - same as above but no noise
        add_noise = False
        # deal with hiddens_hook exactly as above.
        if not self.hiddens_flag:
            p_X_chain_recon, recon_hiddens = self.build_gsn(add_noise=add_noise)
        else:
            p_X_chain_recon, recon_hiddens = self.build_gsn(hiddens=self.hiddens, add_noise=add_noise, reverse=True)

        ####################
        # Costs and output #
        ####################
        log.debug('Cost w.r.t p(X|...) at every step in the graph for the GSN')
        # use the noisy ones for training cost
        costs          = [self.cost_function(rX, self.X) for rX in p_X_chain]
        self.show_cost = costs[-1]  # for a monitor to show progress
        cost           = numpy.sum(costs)  # THIS IS THE TRAINING COST - RECONSTRUCTION OF OUTPUT FROM NOISY GRAPH

        # use the non-noisy graph for prediction
        gsn_costs_recon = [self.cost_function(rX, self.X) for rX in p_X_chain_recon]
        # another monitor, same as self.show_cost but on the non-noisy graph.
        self.monitor    = gsn_costs_recon[-1]
        # this should be considered the main output of the computation, the sample after the
        # last walkback from the non-noisy graph.
        output     = p_X_chain_recon[-1]
        # these should be considered the model's hidden representation - the hidden representation after
        # the last walkback from the non-noisy graph.
        hiddens    = recon_hiddens

        train_mse = T.mean(T.sqr(p_X_chain[-1] - self.X), axis=0)
        train_mse = T.mean(train_mse)

        mse = T.mean(T.sqr(p_X_chain_recon[-1] - self.X), axis=0)
        mse = T.mean(mse)

        monitors = OrderedDict([('noisy_recon_cost', self.show_cost),
                                ('recon_cost', self.monitor),
                                ('mse', mse),
                                ('train_mse', train_mse)])

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
        self.update_layers(self.network_state_output, visible_pX_chain, add_noise=True, reverse=False)

        #####################################################
        #     Create the run and monitor functions      #
        #####################################################
        log.debug("Compiling functions...")
        t = time.time()

        # doesn't make sense to have this if there is a hiddens_hook
        if not self.hiddens_flag:
            # THIS IS THE MAIN PREDICT FUNCTION - takes in a real matrix and produces the output from the non-noisy
            # computation graph
            log.debug("f_run...")
            self.f_run = function(inputs  = [self.X],
                                  outputs = output,
                                  name    = 'gsn_f_run')

        # this is a helper function - it corrupts inputs when testing the non-noisy graph (aka before feeding the
        # input to f_run)
        log.debug("f_noise...")
        self.f_noise = function(inputs  = [self.X],
                                outputs = salt_and_pepper(self.X, self.input_salt_and_pepper, self.mrg),
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

        return cost, monitors, output, hiddens

    ##############################
    # Computation helper methods #
    ##############################
    def build_gsn(self, add_noise, hiddens=None, reverse=False):
        p_X_chain = []
        # Whether or not to corrupt the visible input X
        if add_noise:
            X_init = salt_and_pepper(self.X, self.input_salt_and_pepper, self.mrg)
        else:
            X_init = self.X

        # if no input hiddens were provided, initialize with zeros
        if hiddens is None:
            # init hiddens with zeros
            hiddens = [X_init]
            if self.tied_weights:
                for w in self.weights_list:
                    hiddens.append(T.zeros_like(T.dot(hiddens[-1], w)))
            else:
                for w in self.weights_list[:self.layers]:
                    hiddens.append(T.zeros_like(T.dot(hiddens[-1], w)))

        # The layer update scheme
        log.info("Building the GSN graph : %s updates", str(self.walkbacks))
        for i in range(self.walkbacks):
            log.debug("GSN Walkback %s/%s", str(i + 1), str(self.walkbacks))
            self.update_layers(hiddens, p_X_chain, add_noise, reverse=reverse)
        return p_X_chain, hiddens

    def update_layers(self, hiddens, p_X_chain, add_noise, reverse):
        # in the normal update order, first update odd layers, then even
        if not reverse:
            # One update over the odd layers + one update over the even layers
            log.debug('odd layer updates')
            self.update_odd_layers(hiddens=hiddens, add_noise=add_noise)

            log.debug('even layer updates')
            self.update_even_layers(hiddens=hiddens, p_X_chain=p_X_chain, add_noise=add_noise)

            log.debug('done full update.')
        # otherwise, update even first, then odd (starts from hiddens, ignores starting visible layer)
        else:
            log.debug('even layer updates')
            self.update_even_layers(hiddens=hiddens, p_X_chain=p_X_chain, add_noise=add_noise)

            log.debug('odd layer updates')
            self.update_odd_layers(hiddens=hiddens, add_noise=add_noise)

            log.debug('done full update.')

    def update_odd_layers(self, hiddens, add_noise):
        # Odd layer update function
        # just a loop over the odd layers
        for i in range(1, len(hiddens), 2):
            log.debug('updating layer %s', str(i))
            self.update_single_layer(hiddens=hiddens, p_X_chain=None, layer_idx=i, add_noise=add_noise)

    def update_even_layers(self, hiddens, p_X_chain, add_noise):
        # Even layer update
        # p_X_chain is given to append the p(X|...) at each full update (one update = odd update + even update)
        for i in range(0, len(hiddens), 2):
            log.debug('updating layer %s', str(i))
            self.update_single_layer(hiddens=hiddens, p_X_chain=p_X_chain, layer_idx=i, add_noise=add_noise)

    def update_single_layer(self, hiddens, p_X_chain, layer_idx, add_noise):
        # Compute the dot product, whatever layer
        # If the visible layer X
        if layer_idx == 0:
            if self.tied_weights:
                log.debug('using ' + str(self.weights_list[layer_idx]) + '.T')
                hiddens[layer_idx] = T.dot(hiddens[layer_idx+1], self.weights_list[layer_idx].T) + \
                                     self.bias_list[layer_idx]
            else:
                log.debug('using ' + str(self.weights_list[-(layer_idx+1)]))
                hiddens[layer_idx] = T.dot(hiddens[layer_idx+1], self.weights_list[-(layer_idx+1)]) + \
                                     self.bias_list[layer_idx]
        # If the top layer
        elif layer_idx == len(hiddens) - 1:
            log.debug('using ' + str(self.weights_list[layer_idx-1]))
            hiddens[layer_idx] = T.dot(hiddens[layer_idx-1], self.weights_list[layer_idx-1]) + self.bias_list[layer_idx]
        # Otherwise in-between layers
        else:
            if self.tied_weights:
                log.debug("using %s and %s.T", str(self.weights_list[layer_idx-1]), str(self.weights_list[layer_idx]))
                # next layer        :   hiddens[layer_idx+1], assigned weights : W_i
                # previous layer    :   hiddens[layer_idx-1], assigned weights : W_(layer_idx-1)
                hiddens[layer_idx] = T.dot(hiddens[layer_idx+1], self.weights_list[layer_idx].T) + \
                                     T.dot(hiddens[layer_idx-1], self.weights_list[layer_idx-1]) + \
                                     self.bias_list[layer_idx]
            else:
                log.debug("using %s and %s",
                          str(self.weights_list[layer_idx-1]), str(self.weights_list[-(layer_idx+1)]))
                hiddens[layer_idx] = T.dot(hiddens[layer_idx+1], self.weights_list[-(layer_idx+1)]) + \
                                     T.dot(hiddens[layer_idx-1], self.weights_list[layer_idx-1]) + \
                                     self.bias_list[layer_idx]

        # Add pre-activation noise if NOT input layer
        if layer_idx == 1 and self.noiseless_h1:
            log.debug('>>NO noise in first hidden layer')
            add_noise = False

        # pre activation noise
        if layer_idx != 0 and add_noise:
            log.debug('Adding pre-activation gaussian noise for layer %s', str(layer_idx))
            hiddens[layer_idx] = add_gaussian(hiddens[layer_idx], std=self.hidden_add_noise_sigma, mrg=self.mrg)

        # ACTIVATION!
        if layer_idx == 0:
            log.debug('Activation for visible layer')
            hiddens[layer_idx] = self.visible_activation(hiddens[layer_idx])
        else:
            log.debug('Hidden units activation for layer %s', str(layer_idx))
            hiddens[layer_idx] = self.hidden_activation(hiddens[layer_idx])

        # post activation noise
        # why is there post activation noise? Because there is already pre-activation noise,
        # this just doubles the amount of noise between each activation of the hiddens.
        if layer_idx != 0 and add_noise:
            log.debug('Adding post-activation gaussian noise for layer %s', str(layer_idx))
            hiddens[layer_idx] = add_gaussian(hiddens[layer_idx], std=self.hidden_add_noise_sigma, mrg=self.mrg)

        # build the reconstruction chain if updating the visible layer X
        if layer_idx == 0:
            # if input layer -> append p(X|H...)
            p_X_chain.append(hiddens[layer_idx])

            # sample from p(X|H...) - SAMPLING NEEDS TO BE CORRECT FOR INPUT TYPES
            # I.E. FOR BINARY MNIST SAMPLING IS BINOMIAL. real-valued inputs should be gaussian
            if self.input_sampling:
                if not is_binary(self.visible_activation):
                    # TODO: implement non-binary sampling (gaussian)
                    log.error("Non-binary visible activation sampling not yet supported.")
                    raise NotImplementedError("Non-binary visible activation sampling not yet supported.")
                log.debug('Sampling from input')
                sampled = self.visible_sampling(p=hiddens[layer_idx],
                                                size=hiddens[layer_idx].shape,
                                                dtype=theano.config.floatX)
            else:
                log.debug('>>NO input sampling')
                sampled = hiddens[layer_idx]
            # add noise to input layer
            sampled = salt_and_pepper(sampled, self.input_salt_and_pepper, self.mrg)

            # set input layer
            hiddens[layer_idx] = sampled

    def save_samples(self, initial=None, n_samples=10000):
        log.info('Generating %d samples', n_samples)
        if initial is None:
            initial = numpy.zeros(shape=(1, self.input_size), dtype=theano.config.floatX)
        samples, _ = self.sample(initial=initial, n_samples=n_samples, k=1)
        f_samples = os.path.join(self.outdir, 'gsn_samples.npy')
        numpy.save(f_samples, samples)
        log.debug('saved samples')

    def sample(self, initial, n_samples=400, k=1):
        log.debug("Starting sampling...")
        def sample_some_numbers_single_layer(n_samples):
            x0 = initial
            samples = [x0]
            x = self.f_noise(x0)
            for _ in xrange(n_samples-1):
                x = self.f_sample(x)
                samples.append(x)
                x = self.visible_sampling(n=1, p=x, size=x.shape).astype(theano.config.floatX)
                x = self.f_noise(x)

            log.debug("Sampling done.")
            return numpy.vstack(samples), None

        def sampling_wrapper(NSI):
            # * is the "splat" operator: It takes a list as input, and expands it into actual
            # positional arguments in the function call.
            out = self.f_sample(*NSI)
            NSO = out[:len(self.network_state_output)]
            vis_pX_chain = out[len(self.network_state_output):]
            return NSO, vis_pX_chain

        def sample_some_numbers(n_samples):
            # The network's initial state
            init_vis       = initial
            noisy_init_vis = self.f_noise(init_vis)

            network_state  = [
                [noisy_init_vis] +
                [
                    numpy.zeros(shape=(initial.shape[0], self.layer_sizes[i+1]), dtype=theano.config.floatX)
                    for i in range(len(self.bias_list[1:]))
                ]
            ]

            visible_chain  = [init_vis]
            noisy_h0_chain = [noisy_init_vis]
            sampled_h = []

            times = []
            for i in xrange(n_samples-1):
                _t = time.time()

                # feed the last state into the network, run new state, and obtain visible units expectation chain
                net_state_out, vis_pX_chain = sampling_wrapper(network_state[-1])

                # append to the visible chain
                visible_chain += vis_pX_chain

                # append state output to the network state chain
                network_state.append(net_state_out)

                noisy_h0_chain.append(net_state_out[0])

                if i%k == 0:
                    sampled_h.append(T.stack(net_state_out[1:]))
                    if i == k:
                        log.debug("About "+make_time_units_string(numpy.mean(times)*(n_samples-1-i))+" remaining...")

                times.append(time.time() - _t)

            log.DEBUG("Sampling done.")
            return numpy.vstack(visible_chain), sampled_h

        if self.layers == 1:
            return sample_some_numbers_single_layer(n_samples)
        else:
            return sample_some_numbers(n_samples)

    def create_reconstruction_image(self, input_data):
        """
        Adds noise to an input and saves an image from the reconstruction
        """
        n_examples = len(input_data)
        xs_test = input_data
        noisy_xs_test = self.f_noise(input_data)
        reconstructed = self.run(noisy_xs_test)
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

        save_path = os.path.join(self.outdir, 'gsn_reconstruction.png')
        save_path = os.path.realpath(save_path)
        number_reconstruction.save(save_path)
        log.info("saved output image to %s", save_path)

    @staticmethod
    def pack_hiddens(hiddens_list):
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
        h_list = [T.zeros(shape=(hiddens_tensor.shape[0], self.input_size), dtype=theano.config.floatX)]
        for idx in range(self.layers):
            # we only care about the odd layers
            # (where h0 is the input layer - which makes it even here in the hidden layer space)
            if (idx % 2) != 0:
                h_list.append(
                    T.zeros(shape=(hiddens_tensor.shape[0], self.layer_sizes[idx+1]), dtype=theano.config.floatX)
                )
            else:
                h_list.append(
                    (hiddens_tensor.T[(idx/2)*self.layer_sizes[idx] : (idx/2 + 1)*self.layer_sizes[idx+1]]).T
                )

        return h_list

    ###################
    # Model functions #
    ###################
    def get_inputs(self):
        return [self.X]

    def get_hiddens(self):
        return GSN.pack_hiddens(self.hiddens)

    def get_outputs(self):
        return self.output

    def run(self, input):
        if hasattr(self, 'f_run'):
            # because we use the splat to account for multiple inputs to the function, make sure input is a list.
            input = raise_to_list(input)
            return self.f_run(*input)
        else:
            log.warning("No f_run for the GSN (this is most likely the case when a hiddens_hook was provided.")
            return None

    def get_train_cost(self):
        return self.cost

    def get_monitors(self):
        return self.monitors

    def get_decay_params(self):
        # noise scheduling
        noise_schedule = get_decay_function(self.noise_decay,
                                            self.input_salt_and_pepper,
                                            self.args.get('input_salt_and_pepper'),
                                            self.noise_annealing)
        return [noise_schedule]

    def get_params(self):
        return self.params

    def save_params(self, param_file):
        super(GSN, self).save_params(param_file)

    def save_args(self, args_file="gsn_config.pkl"):
        super(GSN, self).save_args(args_file)