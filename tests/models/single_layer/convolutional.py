# standard libraries
import unittest
# third party
import numpy as np
from theano.tensor import ftensor3, ftensor4
# internal references
from opendeep import function
from opendeep.log.logger import config_root_logger
from opendeep.models import Conv1D, Conv2D


class TestConv(unittest.TestCase):

    def setUp(self):
        # configure the root logger
        # config_root_logger()
        pass

    def testConv1DOutputSize(self):
        try:
            x = ftensor3('x')
            #batch, channels, dim
            s = (None, 15, 94)
            filters = 25
            filter_size = 2
            padding = 2
            stride = 2
            conv1 = Conv1D(inputs=(s, x), n_filters=filters, filter_size=filter_size, padding=padding, stride=stride,
                           outdir=None)
            f1 = function(inputs=[x], outputs=conv1.get_outputs().shape, allow_input_downcast=True)
            x1 = np.ones((100, 15, 94))
            outs = f1(x1)
            self.compareSizes(outs=outs, output_size=conv1.output_size, in_size=s, batches=100)
        finally:
            if 'x' in locals():
                del x
            if 'conv1' in locals():
                del conv1
            if 'f1' in locals():
                del f1
            if 'outs' in locals():
                del outs
            if 'x1' in locals():
                del x1

    def testConv2DOutputSize(self):
        try:
            x = ftensor4('x')
            # batch, channels, height, width
            s = (None, 3, 25, 32)
            filters = 25
            filter_size = 5
            padding = 3
            stride = 3
            conv1 = Conv2D(inputs=(s, x), n_filters=filters, filter_size=filter_size, padding=padding, stride=stride,
                           outdir=None)
            f1 = function(inputs=[x], outputs=conv1.get_outputs().shape, allow_input_downcast=True)
            x1 = np.ones((100, 3, 25, 32))
            outs = f1(x1)
            self.compareSizes(outs=outs, output_size=conv1.output_size, in_size=s, batches=100)

        finally:
            if 'x' in locals():
                del x
            if 'conv1' in locals():
                del conv1
            if 'f1' in locals():
                del f1
            if 'outs' in locals():
                del outs
            if 'x1' in locals():
                del x1

    def compareSizes(self, outs, output_size, in_size, batches):
        self.assertEqual(output_size[0], in_size[0])
        sizes_same = all(np.equal(output_size[1:], outs[1:]))
        self.assertTrue(sizes_same,
                        "Found shapes {!s} (theoretical) and {!s} (computed)".format(output_size[1:], outs[1:]))
        self.assertEqual(outs[0], batches)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
