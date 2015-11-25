"""
This module defines flattening layer (reduce dimensionality)
"""
from __future__ import division
from numpy import prod
# theano imports
from theano.tensor import flatten
# internal references
from opendeep.models.utils import ModifyLayer

class Flatten(ModifyLayer):
    """
    Flattens the trailing dimensions of the expression so that the output has the specified number of dimensions.
    See Theano's `theano.tensor.flatten` operation for implementation details.
    """
    def __init__(self, inputs=None, ndim=1):
        """
        Parameters
        ----------
        inputs : tuple(shape, `Theano.TensorType`)
            tuple(shape, `Theano.TensorType`) or None describing the input to use for this layer.
            `shape` will be a monad tuple representing known sizes for each dimension in the `Theano.TensorType`.
            If 4D images as input, expect formatted as (batch_size, #channels, rows, cols).
        ndim : int
            The number of dimensions for the result to have. (Default 1).
        """
        super(Flatten, self).__init__(inputs=inputs, ndim=ndim)
        input_shape, self.input = self.inputs[0]
        in_ndim = len(input_shape)
        assert 0 < ndim <= in_ndim, \
            "Number of resulting dimensions ndim has to be greater than zero and less than current dims."

        kept_size = tuple(input_shape[:ndim-1])
        flat_size = (None, ) if None in input_shape[ndim-1:] else (prod(input_shape[ndim-1:]), )
        self.output_size = kept_size + flat_size
        self.output = flatten(self.input, ndim)

    def get_inputs(self):
        """
        This should return the input(s) to the layer's computation graph as a list.

        Returns
        -------
        Theano variable
            Theano variables representing the input to the layer's computation.
        """
        return self.input

    def get_outputs(self):
        """
        This method will return the layer's output variable expression from the computational graph.

        This will be used for creating hooks to link models together,
        where these outputs can be strung as the inputs or hiddens to another model :)

        Returns
        -------
        theano expression
            Theano expression for the flattened input.

        """
        return self.output
