"""
This module defines pooling layers (like MaxPooling used in convolutional nets).
"""
from __future__ import division
# theano imports
try:
    from theano.tensor.signal.pool import pool_2d
except ImportError:
    from theano.tensor.signal.downsample import max_pool_2d as pool_2d
# internal references
from opendeep.models.utils import ModifyLayer

# flag for having NVIDIA's CuDNN library.
has_cudnn = True
try:
    from theano.sandbox.cuda.dnn import dnn_pool, dnn_available
    has_cudnn = dnn_available()
except ImportError as e:
    has_cudnn = False

def _pool_out_size(imgshape, ds, st, padding, ignore_border=True):
    if st is None:
        st = ds

    assert len(ds) == len(st), "stride and size need to have the same number of dimensions"
    ndims = len(ds)
    pooldims = list(imgshape[-ndims:])
    pooldims = [dim + pad*2 if dim is not None and pad is not None
                else None
                for dim, pad in zip(pooldims, padding)]

    if ignore_border:
        outdims = [(dim - size) // stride + 1
                   if dim is not None and size is not None and stride is not None
                   else None
                   for dim, size, stride in zip(pooldims, ds, st)]
        outdims = [max(outdim, 0)
                   if outdim is not None
                   else None
                   for outdim in outdims]
    else:
        outdims = []
        for dim, size, stride in zip(pooldims, ds, st):
            if dim is not None and size is not None and stride is not None:
                if stride >= size:
                    outdims.append((dim - 1) // stride + 1)
                else:
                    outdims.append(max(0, (dim - 1 - size) // stride + 1) + 1)
            else:
                outdims.append(None)

    rval = list(imgshape[:-ndims]) + outdims
    return rval


class Pool2D(ModifyLayer):
    """
    Performs pooling over the last 2 dimensions of the input. If 4D input (images), expect
    form (batch, channel, rows, cols).
    """
    def __init__(self, inputs=None, size=(1, 1), stride=None, pad=(0, 0), mode='max', ignore_border=True):
        """
        Parameters
        ----------
        inputs : tuple(shape, `Theano.TensorType`)
            tuple(shape, `Theano.TensorType`) or None describing the input to use for this layer.
            `shape` will be a monad tuple representing known sizes for each dimension in the `Theano.TensorType`.
            If 4D images as input, expect formatted as (batch_size, #channels, rows, cols).
        size : tuple(int) or int
            Downsample factor over (rows, columns). If it is an int, it will be the same size for rows and cols.
        stride : tuple(int) or int
            Stride size (step size), which is the number of shifts over rows/cols to get the
            next pool region. If it is an int, it will be the same size for rows and cols.
        pad : tuple(int) or int
            (pad_h, pad_w), pad zeros to extend beyond four borders
            of the images, pad_h is the size of the top and bottom margins,
            and pad_w is the size of the left and right margins. If it is an int, it will be the same
            size for rows and cols.
        mode : 'max', 'sum', 'average_inc_pad', 'average_exc_pad'
            Operation executed on each window. `max` and `sum` always exclude
            the padding in the computation. `average` gives you the choice to
            include or exclude it.
        ignore_border : bool
            If `size` doesn't divide the input `shape`, do we include an extra row/col of
            partial downsampling (False) or ignore it (True). When True, (5,5) input with size=(2,2)
            will generate a (2,2) output. (3,3) otherwise.
        """
        super(Pool2D, self).__init__(inputs=inputs, size=size, stride=stride, pad=pad,
                                     mode=mode, ignore_border=ignore_border)
        input_shape, self.input = self.inputs[0]
        if isinstance(size, int):
            size = (size, ) * 2
        if stride is None:
            stride = size
        if isinstance(stride, int):
            stride = (stride, ) * 2
        if isinstance(pad, int):
            pad = (pad, ) * 2

        assert len(size) == len(stride) == len(pad), "Size, stride, and pad must have the same number of dimensions."

        self.output_size = tuple(_pool_out_size(imgshape=input_shape,
                                                ds=size,
                                                st=stride,
                                                ignore_border=ignore_border,
                                                padding=pad))

        cudnn_modes = ['max', 'average_inc_pad', 'average_exc_pad']
        if has_cudnn and mode in cudnn_modes and ignore_border and self.input.ndim == 4:
            self.output = dnn_pool(img=self.input,
                                   ws=size,
                                   stride=stride,
                                   mode=mode,
                                   pad=pad)
        else:
            self.output = pool_2d(input=self.input,
                                  ds=size,
                                  st=stride,
                                  padding=pad,
                                  mode=mode,
                                  ignore_border=ignore_border)

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
            Theano expression for the pooled (downsampled) input.

        """
        return self.output
