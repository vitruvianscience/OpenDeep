"""
This module defines noise to add in the network.
"""
# standard imports
import logging
# third pary imports
import theano.sandbox.rng_mrg as RNG_MRG
from theano.tensor import switch as Tswitch
# internal references
from opendeep.models.utils import ModifyLayer
from opendeep.utils.constructors import sharedX
from opendeep.utils.noise import get_noise
from opendeep.utils.decay import get_decay_function

log = logging.getLogger(__name__)

class Noise(ModifyLayer):
    """
    The :class:`Noise` is a layer that adds different noise functions to the computation graph, e.g.
    Dropout, Gaussian, Uniform, Salt-And-Pepper, etc.
    """
    def __init__(self, inputs=None,
                 noise='dropout', noise_level=0.5, noise_decay=False, noise_decay_amount=0.99,
                 mrg=RNG_MRG.MRG_RandomStreams(1), switch=True):
        """
        Parameters
        ----------
        inputs : tuple(shape, `Theano.TensorType`)
            tuple(shape, `Theano.TensorType`) describing the inputs to use for this layer.
            `shape` will be a monad tuple representing known sizes for each dimension in the `Theano.TensorType`.
            The length of `shape` should be equal to number of dimensions in `Theano.TensorType`, where the shape
            element is an integer representing the size for its dimension, or None if the shape isn't known.
            For example, if you have a matrix with unknown batch size but fixed feature size of 784, `shape` would
            be: (None, 784). The full form of `inputs` would be:
            [((None, 784), <TensorType(float32, matrix)>)].
        noise : str
            What type of noise to use for the output. See opendeep.utils.noise
            for options. This should be appropriate for the unit activation, i.e. Gaussian for tanh or other
            real-valued activations, etc.
        noise_level : float
            The amount of noise to use for the noise function specified by `noise`. This could be the
            standard deviation for gaussian noise, the interval for uniform noise, the dropout amount, etc.
        noise_decay : str or False
            Whether to use `noise` scheduling (decay `noise_level` during the course of training),
            and if so, the string input specifies what type of decay to use. See opendeep.utils.decay for options.
            Noise decay (known as noise scheduling) effectively helps the model learn larger variance features first,
            and then smaller ones later (almost as a kind of curriculum learning). May help it converge faster.
        noise_decay_amount : float
            The amount to reduce the `noise_level` after each training epoch based on the decay function specified
            in `noise_decay`.
        mrg : random
            A random number generator that is used when adding noise.
            I recommend using Theano's sandbox.rng_mrg.MRG_RandomStreams.
        switch : boolean
            Whether to create a switch to turn noise on during training and off during testing (True). If False,
            noise will be applied at both training and testing times.
        """
        super(Noise, self).__init__(inputs=inputs, outputs=inputs[0],
                                    noise=noise, noise_level=noise_level,
                                    noise_decay=noise_decay, noise_decay_amount=noise_decay_amount,
                                    mrg=mrg, switch=switch)
        # self.inputs is a list from superclass initialization, grab the first element
        self.inputs = self.inputs[0][1]
        log.debug('Adding %s noise switch.' % str(noise))
        if noise_level is not None:
            noise_level = sharedX(value=noise_level)
            noise_func = get_noise(noise, noise_level=noise_level, mrg=mrg)
        else:
            noise_func = get_noise(noise, mrg=mrg)

        # apply the noise as a switch!
        # default to apply noise. this is for the cost and gradient functions to be computed later
        # (not sure if the above statement is accurate such that gradient depends on initial value of switch)
        if switch:
            self.noise_switch = sharedX(value=1, name="noise_switch")

        # noise scheduling
        if noise_decay and noise_level is not None:
            self.noise_schedule = get_decay_function(noise_decay,
                                                     noise_level,
                                                     noise_level.get_value(),
                                                     noise_decay_amount)
        # apply noise to the inputs!
        if switch:
            self.outputs = Tswitch(self.noise_switch,
                                   noise_func(input=self.inputs),
                                   self.inputs)
        else:
            self.outputs = noise_func(input=self.inputs)

    def get_inputs(self):
        """
        This should return the input to the layer's computation graph.

        Returns
        -------
        Theano variable
            Theano variable representing the input(s) to the layer's computation.
        """
        return self.inputs

    def get_outputs(self):
        """
        This method will return the layer's output variable expression from the computational graph.

        Returns
        -------
        theano expression
            Theano expression of the output from the activation function.
        """
        return self.outputs

    def get_decay_params(self):
        """
        This method returns any noise decay function for noise scheduling during training.

        Returns
        -------
        list(:class:`DecayFunction`)
            List of opendeep.utils.decay_functions.DecayFunction objects of the parameters to decay for this model.
        """
        if hasattr(self, 'noise_schedule'):
            # noise scheduling
            return [self.noise_schedule]
        else:
            return []

    def get_switches(self):
        """
        This method returns the noise switch to turn on during training and off during testing.

        Returns
        -------
        list
            List containing the noise switch. (or empty list if switch is False)
        """
        if hasattr(self, 'noise_switch'):
            # noise switch
            return [self.noise_switch]
        else:
            return []
