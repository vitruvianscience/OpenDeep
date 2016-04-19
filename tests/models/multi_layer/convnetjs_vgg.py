import unittest
import numpy as np
import theano
from theano.tensor import tensor4, matrix, mean, neq, lvector
from opendeep import config_root_logger
from opendeep.data import ModifyStream
from opendeep.models import Prototype, Conv2D, Softmax
from opendeep.models.utils import Pool2D, Flatten
from opendeep.monitor import Monitor
from opendeep.optimization.loss import Neg_LL
from opendeep.optimization import AdaDelta
from opendeep.data import CIFAR10


class TestParamSharing(unittest.TestCase):

    def setUp(self):
        # configure the root logger
        # config_root_logger()
        pass

    def testCifar(self):
        data, cifar, loss, optimizer = [None]*4
        try:
            data = CIFAR10(one_hot=True, path='../datasets/cifar-10-batches-py/')

            # augment by flipping and shifting
            def shift(img):
                dx = np.floor(np.random.rand * 5 - 2)
                dy = np.floor(np.random.rand * 5 - 2)
                if dx != 0. and dy != 0.:
                    _img = np.zeros(img.size)
                    for dim in img:
                        for row in dim:
                            for col in row:
                                if 0 < row+dy < _img.size[1] and 0 < col+dx < _img.size[2]:
                                    _img[dim, row+dy, col+dx] = img[dim, row, col]
                else:
                    return img

            def flip_horiz(img, cutoff=.5):
                if np.random.rand < cutoff:
                    return img[:, :, ::-1]
                else:
                    return img

            cifar = Prototype()
            x = ((None, 3, 32, 32), tensor4('x'))

            cifar.add(
                Conv2D(inputs=x, n_filters=16, filter_size=5, stride=1, pad=2, activation='relu')
            )
            cifar.add(
                Pool2D, size=2, stride=2
            )
            cifar.add(
                Conv2D, n_filters=20, filter_size=5, stride=1, pad=2, activation='relu'
            )
            cifar.add(
                Pool2D, size=2, stride=2
            )
            cifar.add(
                Conv2D, n_filters=20, filter_size=5, stride=1, pad=2, activation='relu'
            )
            cifar.add(
                Flatten, ndim=2
            )
            cifar.add(
                Softmax, outputs=10, out_as_probs=True
            )

            y = matrix('y')
            loss = Neg_LL(inputs=cifar.get_outputs(), targets=y, one_hot=True)

            optimizer = AdaDelta(model=cifar,
                                 dataset=data,
                                 loss=loss,
                                 epochs=10,
                                 batch_size=4,
                                 learning_rate=.01)
            optimizer.train()
        finally:
            del data, cifar, loss, optimizer

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
