'''
.. module:: autoencoder

This module contains different autoencoders to use.

A Denoising Autoencoder is a special case of Generative Stochastic Networks. It is used as a basic,
unsupervised, generative building block for feedforward networks.

'Generalized Denoising Auto-Encoders as Generative Models'
Yoshua Bengio, Li Yao, Guillaume Alain, Pascal Vincent
http://papers.nips.cc/paper/5023-generalized-denoising-auto-encoders-as-generative-models.pdf
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
import theano.sandbox.rng_mrg as RNG_MRG
# internal references
from opendeep.models.model import Model
from opendeep.models.multi_layer.generative_stochastic_network import GSN

log = logging.getLogger(__name__)

class DenoisingAutoencoder(GSN):
    '''
    Class for creating a new Denoising Autoencoder (DAE)
    This is a special case of a GSN with only one hidden layer
    '''
    # Default values to use for some DAE parameters
    _defaults = {# gsn parameters
                "walkbacks": 1,
                "input_size": None,  # number of input units - please specify for your dataset!
                "hidden_size": 1500,
                "visible_activation": 'sigmoid',
                "hidden_activation": 'tanh',
                "input_sampling": True,
                "MRG": RNG_MRG.MRG_RandomStreams(1),
                "weights_init": "uniform",  # how to initialize weights
                'weights_interval': 'montreal',  # if the weights_init was 'uniform', how to initialize from uniform
                'weights_mean': 0,  # mean for gaussian weights init
                'weights_std': 0.005,  # standard deviation for gaussian weights init
                'bias_init': 0.0,  # how to initialize the bias parameter
                # train param
                "cost_function": 'binary_crossentropy',
                # noise parameters
                "noise_decay": 'exponential',  # noise schedule algorithm
                "noise_annealing": 1.0, #no noise schedule by default
                "add_noise": True,
                "noiseless_h1": True,
                "hidden_add_noise_sigma": 2,
                "input_salt_and_pepper": 0.4,
                # data parameters
                "output_path": 'outputs/dae/',
                "is_image": True,
                "vis_init": False}

    def __init__(self, config=None, defaults=_defaults, inputs_hook=None, hiddens_hook=None, dataset=None,
                 walkbacks=None, input_size=None, hidden_size=None, visible_activation=None, hidden_activation=None,
                 input_sampling=None, MRG=None, weights_init=None, weights_interval=None, weights_mean=None,
                 weights_std=None, bias_init=None, cost_function=None, noise_decay=None, noise_annealing=None,
                 add_noise=None, noiseless_h1=None, hidden_add_noise_sigma=None, input_salt_and_pepper=None,
                 output_path=None, is_image=None, vis_init=None):
        # init Model
        # force the model to have one layer - DAE is a specific GSN with a single hidden layer
        defaults['layers'] = 1
        if config:
            config['layers'] = 1
        # init Model to combine the defaults and config dictionaries with the initial parameters.
        super(DenoisingAutoencoder, self).__init__(
            **{arg: val for (arg, val) in locals().iteritems() if arg is not 'self'}
        )
        # all configuration parameters are now in self!

class ContractiveAutoencoder(Model):
    '''
    A contractive autoencoder
    https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/cA.py
    '''
    # TODO: ContractiveAutoencoder
    def __init__(self):
        super(ContractiveAutoencoder, self).__init__()
        log.error("ContractiveAutoencoder not implemented yet!")
        raise NotImplementedError("ContractiveAutoencoder not implemented yet!")

class SparseAutoencoder(Model):
    '''
    A sparse autoencoder
    '''
    # TODO: SparseAutoencoder
    def __init__(self):
        super(SparseAutoencoder, self).__init__()
        log.error("SparseAutoencoder not implemented yet!")
        raise NotImplementedError("SparseAutoencoder not implemented yet!")

class StackedDAE(Model):
    '''
    A stacked Denoising Autoencoder stacks multiple layers of DAE's
    '''
    # TODO: Stacked DAE
    def __init__(self):
        super(StackedDAE, self).__init__()
        log.error("StackedDAE not implemented yet!")
        raise NotImplementedError("StackedDAE not implemented yet!")