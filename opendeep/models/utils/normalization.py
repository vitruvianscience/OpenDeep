"""
This module defines pooling layers (like MaxPooling used in convolutional nets).
"""
from __future__ import division
# theano imports
from theano.tensor import (sqr, alloc, set_subtensor)
from theano.tensor.nnet.bn import batch_normalization
# internal references
from opendeep.models.utils import ModifyLayer


class LRN(ModifyLayer):
    """
    Performs cross-channel local response normalization for 2D feature maps.

    implementation from Lasagne: https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/normalization.py

    Aggregation is purely across channels, not within channels,
    and performed "pixelwise".
    Input order is assumed to be `BC01`.
    If the value of the ith channel is :math:`x_i`, the output is

    .. math::
        x_i = \frac{x_i}{ (k + ( \alpha \sum_j x_j^2 ))^\beta }

    where the summation is performed over this position on :math:`n`
    neighboring channels.
    """
    def __init__(self, inputs=None, alpha=1e-4, k=2, beta=0.75, n=5):
        """
        Parameters
        ----------
        inputs : tuple(shape, `Theano.TensorType`)
            tuple(shape, `Theano.TensorType`) or None describing the input to use for this layer.
            `shape` will be a monad tuple representing known sizes for each dimension in the `Theano.TensorType`.
            If 4D images as input, expect formatted as (batch_size, #channels, rows, cols).
        alpha : float
            alpha term for the LRN equation
        k : int
            k term for LRN equation
        beta : float
            beta term for LRN equation
        n : int
            The number of adjacent channels to normalize over
        """
        super(LRN, self).__init__(inputs=inputs, alpha=alpha, k=k, beta=beta, n=n)
        input_shape, self.input = self.inputs[0]

        # output same shape as input
        self.output_size = input_shape

        half_n = n // 2
        input_sqr = sqr(self.input)

        if any([s is None for s in input_shape]):
            input_shape = self.input.shape
        b, ch, r, c = input_shape
        extra_channels = alloc(0., b, ch + 2 * half_n, r, c)
        input_sqr = set_subtensor(extra_channels[:, half_n:half_n + ch, :, :], input_sqr)
        scale = k
        for i in range(n):
            scale += alpha * input_sqr[:, i:i + ch, :, :]
        scale = scale ** beta
        self.output = self.input / scale

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
            Theano expression for the normalized input.

        """
        return self.output

class BatchNorm(ModifyLayer):
    """
    Applies Batch Normalization as defined by:
    "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
    Sergey Ioffe, Christian Szegedy
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, inputs=None, gamma=None, beta=None, mean=None, std=None, mode='low_mem'):
        """
        Parameters
        ----------
        inputs : symbolic tensor
            Mini-batch of activations
        gamma : symbolic tensor
            BN scale parameter, must be of same dimensionality as inputs and broadcastable against it
        beta : symbolic tensor
            BN shift parameter, must be of same dimensionality as inputs and broadcastable against it
        mean : symbolic tensor
            inputs means, must be of same dimensionality as inputs and broadcastable against it
        std : symbolic tensor
            inputs standard deviation, must be of same dimensionality as inputs and broadcastable against it
        mode : string
            'low_mem' or 'high_mem' Specify which batch_normalization implementation that will be used.
            As no intermediate representations are stored for the back-propagation, 'low_mem' implementation lower
            the memory usage, however, it is 5-10% slower than 'high_mem' implementation. Note that 5-10% computation
            time difference compare the batch_normalization operation only, time difference between implementation is
            likely to be less important on the full model fprop/bprop.

        """
        self.input=None
        self.output=None
        raise NotImplementedError("Batch Normalization not yet implemented!")

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
            Theano expression for the normalized input.

        """
        return self.output
