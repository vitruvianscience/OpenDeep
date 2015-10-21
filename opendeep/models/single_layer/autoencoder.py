"""
This module contains different autoencoders to use. Autoencoders are nonprobabilitic, generative models
that try to reconstruct an input from a hidden layer.
input <-> hidden, or laid out, X -> H -> X
"""
# standard libraries
import logging
import theano.sandbox.rng_mrg as RNG_MRG
# internal references
from opendeep.models.multi_layer.generative_stochastic_network import GSN

log = logging.getLogger(__name__)


class DenoisingAutoencoder(GSN):
    """
    Class for creating a new Denoising Autoencoder (DAE).

    A Denoising Autoencoder is a special case of Generative Stochastic Networks with only one hidden layer.
    It is used as a basic, unsupervised, generative building block for deep networks.

    'Generalized Denoising Auto-Encoders as Generative Models'
    Yoshua Bengio, Li Yao, Guillaume Alain, Pascal Vincent
    http://papers.nips.cc/paper/5023-generalized-denoising-auto-encoders-as-generative-models.pdf

    Scheduled noise is added as discussed in the paper:
    'Scheduled denoising autoencoders'
    Krzysztof J. Geras, Charles Sutton
    http://arxiv.org/abs/1406.3269
    """
    def __init__(self, inputs=None, hiddens=1000, params=None, outdir='outputs/dae/',
                 visible_activation='sigmoid', hidden_activation='tanh',
                 walkbacks=1,
                 input_sampling=True, mrg=RNG_MRG.MRG_RandomStreams(1),
                 tied_weights=True,
                 weights_init='uniform', weights_interval='montreal', weights_mean=0, weights_std=5e-3,
                 bias_init=0.0,
                 add_noise=True, noiseless_h1=True,
                 hidden_noise='gaussian', hidden_noise_level=2, input_noise='salt_and_pepper', input_noise_level=0.4,
                 noise_decay='exponential', noise_annealing=1,
                 image_width=None, image_height=None):
        """
        Initialize a DAE.

        Parameters
        ----------
        inputs : tuple(shape, `Theano.TensorType`)
            The dimensionality of the inputs for this model, and the routing information for the model
            to accept inputs from elsewhere. `shape` will be a monad tuple representing known
            sizes for each dimension in the `Theano.TensorType`. The length of `shape` should be equal to number of
            dimensions in `Theano.TensorType`, where the shape element is an integer representing the size for its
            dimension, or None if the shape isn't known. For example, if you have a matrix with unknown batch size
            but fixed feature size of 784, `shape` would be: (None, 784). The full form of `inputs` would be:
            [((None, 784), <TensorType(float32, matrix)>)].
        hiddens : tuple(shape, `Theano.TensorType`) or shape or int
            The dimensionality of the hidden representation for this model, or the routing information for
            the model to accept its hidden representation from elsewhere. Generally, you want it to be larger than
            `input_size`, which is known as *overcomplete*.
        params : Dict(string_name: theano SharedVariable), optional
            A dictionary of model parameters (shared theano variables) that you should use when constructing
            this model (instead of initializing your own shared variables). This parameter is useful when you want to
            have two versions of the model that use the same parameters - such as siamese networks or pretraining some
            weights.
        outdir : str
            The directory you want outputs (parameters, images, etc.) to save to. If None, nothing will
            be saved.
        visible_activation : str or callable
            The nonlinear (or linear) visible activation to perform after the dot product from hiddens -> visible layer.
            This activation function should be appropriate for the input unit types, i.e. 'sigmoid' for binary inputs.
            See opendeep.utils.activation for a list of available activation functions. Alternatively, you can pass
            your own function to be used as long as it is callable.
        hidden_activation : str or callable
            The nonlinear (or linear) hidden activation to perform after the dot product from visible -> hiddens layer.
            See opendeep.utils.activation for a list of available activation functions. Alternatively, you can pass
            your own function to be used as long as it is callable.
        walkbacks : int
            The number of walkbacks to perform (the variable K in Bengio's paper above). A walkback is a Gibbs sample
            from the DAE, which means the model generates inputs in sequence, where each generated input is compared
            to the original input to create the reconstruction cost for training. For running the model, the very last
            generated input in the Gibbs chain is used as the output.
        input_sampling : bool
            During walkbacks, whether to sample from the generated input to create a new starting point for the next
            walkback (next step in the Gibbs chain). This generally makes walkbacks more effective by making the
            process more stochastic - more likely to find spurious modes in the model's representation.
        mrg : random
            A random number generator that is used when adding noise into the network and for sampling from the input.
            I recommend using Theano's sandbox.rng_mrg.MRG_RandomStreams.
        tied_weights : bool
            DAE has two weight matrices - W from input -> hiddens and V from hiddens -> input. This boolean
            determines if V = W.T, which 'ties' V to W and reduces the number of parameters necessary during training.
        weights_init : str
            Determines the method for initializing model weights. See opendeep.utils.nnet for options.
        weights_interval : str or float
            If Uniform `weights_init`, the +- interval to use. See opendeep.utils.nnet for options.
        weights_mean : float
            If Gaussian `weights_init`, the mean value to use.
        weights_std : float
            If Gaussian `weights_init`, the standard deviation to use.
        bias_init : float
            The initial value to use for the bias parameter. Most often, the default of 0.0 is preferred.
        add_noise : bool
            Whether to add noise (corrupt) the input before passing it through the computation graph during training.
            This should most likely be set to the default of True, because this is a *denoising* autoencoder after all.
        noiseless_h1 : bool
            Whether to not add noise (corrupt) the hidden layer during computation.
        hidden_noise : str
            What type of noise to use for corrupting the hidden layer (if not `noiseless_h1`). See opendeep.utils.noise
            for options. This should be appropriate for the hidden unit activation, i.e. Gaussian for tanh or other
            real-valued activations, etc.
        hidden_noise_level : float
            The amount of noise to use for the noise function specified by `hidden_noise`. This could be the
            standard deviation for gaussian noise, the interval for uniform noise, the dropout amount, etc.
        input_noise : str
            What type of noise to use for corrupting the input before computation (if `add_noise`).
            See opendeep.utils.noise for options. This should be appropriate for the input units, i.e. salt-and-pepper
            for binary units, etc.
        input_noise_level : float
            The amount of noise used to corrupt the input. This could be the masking probability for salt-and-pepper,
            standard deviation for Gaussian, interval for Uniform, etc.
        noise_decay : str or False
            Whether to use `input_noise` scheduling (decay `input_noise_level` during the course of training),
            and if so, the string input specifies what type of decay to use. See opendeep.utils.decay for options.
            Noise decay (known as noise scheduling) effectively helps the DAE learn larger variance features first,
            and then smaller ones later (almost as a kind of curriculum learning). May help it converge faster.
        noise_annealing : float
            The amount to reduce the `input_noise_level` after each training epoch based on the decay function specified
            in `noise_decay`.
        image_width : int
            If the input should be represented as an image, the width of the input image. If not specified, it will be
            close to the square factor of the `input_size`.
        image_height : int
            If the input should be represented as an image, the height of the input image. If not specified, it will be
            close to the square factor of the `input_size`.
        """
        # force the model to have one layer - DAE is a specific GSN with a single hidden layer
        layers = 1
        # init GSN because DenoisingAutoencoder is a special case with layers=1
        super(DenoisingAutoencoder, self).__init__(
            **{arg: val for (arg, val) in locals().items() if arg is not 'self'}
        )
        # Done, since this is a special case of the GSN! Easy peasy.
