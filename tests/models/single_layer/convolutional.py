# standard libraries
from __future__ import print_function
import unittest
import shutil
# third party
import numpy as np
from theano.tensor import ftensor3, ftensor4, lvector, mean, neq
# internal references
from opendeep import function
from opendeep.log.logger import config_root_logger
from opendeep.models import Conv1D, Conv2D, Prototype, Dense, Softmax
from opendeep.models.utils import Pool2D, Flatten
from opendeep.data import MNIST
from opendeep.monitor import Monitor, FileService
from opendeep.optimization import SGD
from opendeep.optimization.loss import Neg_LL

# python 2 vs. python 3 mnist source
import sys
if sys.version_info > (3, 0):
    mnist_name = 'mnist_py3k.pkl.gz'
else:
    mnist_name = 'mnist.pkl.gz'


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

    def testLeNet(self):
        try:
            # quick and dirty way to create a model from arbitrary layers
            lenet = Prototype(outdir=None)
            # our input is going to be 4D tensor of images with shape (batch_size, 1, 28, 28)
            x = ((None, 1, 28, 28), ftensor4('x'))
            # our first convolutional layer
            lenet.add(
                Conv2D(inputs=x, n_filters=20, filter_size=(5, 5), outdir=None)
            )
            # our first pooling layer, automatically hooking inputs to the previous convolutional outputs
            lenet.add(
                Pool2D, size=(2, 2)
            )
            # our second convolutional layer
            lenet.add(
                Conv2D, n_filters=50, filter_size=(5, 5), outdir=None
            )
            # our second pooling layer
            lenet.add(
                Pool2D, size=(2, 2)
            )
            # now we need to flatten the 4D convolution outputs into 2D matrix (just flatten the trailing dimensions)
            lenet.add(
                Flatten, ndim=2
            )
            # one dense hidden layer
            lenet.add(
                Dense, outputs=500, activation='tanh', outdir=None
            )
            # hook a softmax classification layer, outputting the probabilities.
            lenet.add(
                Softmax, outputs=10, out_as_probs=True, outdir=None
            )

            # Grab the MNIST dataset
            data = MNIST(path="../../../datasets/{!s}".format(mnist_name), concat_train_valid=False, flatten=False)
            # define our loss to optimize for the model (and the target variable)
            # targets from MNIST are int64 numbers 0-9
            y = lvector('y')
            loss = Neg_LL(inputs=lenet.get_outputs(), targets=y, one_hot=False)
            # monitor
            error_monitor = Monitor(name='error', expression=mean(neq(lenet.models[-1].y_pred, y)), valid=True,
                                    test=True, out_service=FileService('outputs/lenet_error.txt'))
            # optimize our model to minimize loss given the dataset using SGD
            optimizer = SGD(model=lenet,
                            dataset=data,
                            loss=loss,
                            epochs=10,
                            batch_size=128,
                            learning_rate=.1,
                            momentum=False)
            print("Training LeNet...")
            optimizer.train(monitor_channels=error_monitor)

            def test_subset(filename, expected, conf=0.001):
                with open(filename, 'r') as f:
                    errs = [float(err) for err in f]
                for i, (err, exp) in enumerate(zip(errs, expected)):
                    if i == 0:
                        c = conf*10
                    else:
                        c = conf
                    self.assertTrue(exp-c < round(err, 4) < exp+c,
                                    "Errors: {!s} and Expected: {!s} -- Error at {!s} and {!s}".format(
                                        errs, expected, err, exp)
                                    )

            test_subset('outputs/lenet_error_train.txt',
                        [.0753,
                         .0239,
                         .0159,
                         .0113,
                         .0088,
                         .0064,
                         .0050,
                         .0037,
                         .0026,
                         .0019]
                        )
            test_subset('outputs/lenet_error_valid.txt',
                        [.0283,
                         .0209,
                         .0170,
                         .0151,
                         .0139,
                         .0129,
                         .0121,
                         .0118,
                         .0112,
                         .0113]
                        )
            test_subset('outputs/lenet_error_test.txt',
                        [.0319,
                         .0213,
                         .0167,
                         .0134,
                         .0122,
                         .0119,
                         .0116,
                         .0107,
                         .0104,
                         .0105]
                        )
            shutil.rmtree('outputs/')

        finally:
            if 'lenet' in locals():
                del lenet
            if 'data' in locals():
                del data
            if 'y' in locals():
                del y
            if 'x' in locals():
                del x
            if 'loss' in locals():
                del loss
            if 'optimizer' in locals():
                del optimizer

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
