"""
This module defines common Activation layers (ReLu, tanh, sigmoid, etc.).
"""
# internal references
from opendeep.models.utils import ModifyLayer
from opendeep.utils.activation import get_activation_function

class Activation(ModifyLayer):
    """
    The :class:`Activation` is a layer that performs the nonlinearity. If you use this layer, it is suggested
    to set the activation parameter for the model beforehand to 'linear', so that you aren't applying an
    activation twice (unless you mean to).
    """
    def __init__(self, inputs=None, activation=None):
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
        activation : str or callable
            The activation function to use going from input -> output. This can be a string
            representing an option from `opendeep.utils.activation`, or your own function as long as it is callable.
        """
        super(Activation, self).__init__(inputs=inputs, activation=activation)
        # self.inputs is a list from superclass initialization, grab the first element
        self.output_size, self.inputs = self.inputs[0]
        # activation function!
        activation_func = get_activation_function(activation)
        self.outputs = activation_func(self.inputs)

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
