"""
Tutorial: Classifying Handwritten MNIST Images using a simple convolutional net (LeNet5)
"""
import numpy as np
from theano.tensor import tensor4, lvector
from opendeep import config_root_logger
from opendeep.data import ModifyStream
from opendeep.models import Prototype, Conv2D, Dense, Softmax
from opendeep.models.utils import Pool2D
from opendeep.optimization.loss import Neg_LL
from opendeep.optimization import SGD
from opendeep.data import MNIST

# grab a log to output useful info
config_root_logger()

def process_mnist(input_image):
    return np.reshape(input_image, (1, 28, 28))

def build_lenet():
    lenet = Prototype()

    x = tensor4('x')
    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor
    # (28, 28) is the size of MNIST images.
    # layer0_input = x.reshape((x.shape[0], 1, 28, 28))

    lenet.add(
        Conv2D(
            inputs=[((None, 1, 28, 28), x)],
            n_filters=20, filter_size=(5, 5)
        )
    )
    lenet.add(
        Pool2D, size=(2, 2)
    )
    lenet.add(
        Conv2D, n_filters=50, filter_size=(5, 5)
    )
    lenet.add(
        Pool2D, size=(2, 2)
    )

    dense_input = lenet.models[-1].get_outputs().flatten(2)
    dense_input_shape = (None, np.prod(lenet.models[-1].output_size[1:]))

    lenet.add(
        Dense(
            inputs=[(dense_input_shape, dense_input)],
            outputs=500,
            activation='tanh'
        )
    )
    lenet.add(
        Softmax, outputs=10, out_as_probs=True
    )

    return lenet


if __name__ == '__main__':
    lenet = build_lenet()
    data = MNIST(concat_train_valid=True)
    data.train_inputs = ModifyStream(data.train_inputs, process_mnist)
    data.valid_inputs = ModifyStream(data.valid_inputs, process_mnist)
    data.test_inputs = ModifyStream(data.test_inputs, process_mnist)
    y = lvector('y')
    loss = Neg_LL(inputs=lenet.get_outputs(), targets=y, one_hot=False)
    optimizer = SGD(model=lenet,
                    dataset=data,
                    loss=loss,
                    epochs=200,
                    batch_size=500,
                    learning_rate=.1,
                    momentum=.9,
                    nesterov_momentum=True)
    optimizer.train()
