"""
Tutorial: Classifying Handwritten MNIST Images using a simple convolutional net (LeNet5)
"""
import numpy as np
from theano.tensor import tensor4, lvector, mean, neq
from opendeep import config_root_logger
from opendeep.data import ModifyStream
from opendeep.models import Prototype, Conv2D, Dense, Softmax
from opendeep.models.utils import Pool2D
from opendeep.monitor import Monitor
from opendeep.optimization.loss import Neg_LL
from opendeep.optimization import SGD
from opendeep.data import MNIST

# grab a log to output useful info
config_root_logger()

def build_lenet():
    # quick and dirty way to create a model from arbitrary layers
    lenet = Prototype()

    # our input is going to be 4D tensor of images with shape (batch_size, 1, 28, 28)
    x = ((None, 1, 28, 28), tensor4('x'))

    # our first convolutional layer
    lenet.add(
        Conv2D(inputs=x, n_filters=20, filter_size=(5, 5))
    )
    # our first pooling layer, automatically hooking inputs to the previous convolutional outputs
    lenet.add(
        Pool2D, size=(2, 2)
    )
    # our second convolutional layer
    lenet.add(
        Conv2D, n_filters=50, filter_size=(5, 5)
    )
    # our second pooling layer
    lenet.add(
        Pool2D, size=(2, 2)
    )

    # now we need to flatten the 4D convolution outputs into 2D matrix (just flatten the trailing dimensions)
    dense_input = lenet.models[-1].get_outputs().flatten(2)
    # redefine the size appropriately for flattening (since we are doing a Theano modification)
    dense_input_shape = (None, np.prod(lenet.models[-1].output_size[1:]))
    # pass this flattened matrix as the input to a Dense layer!
    lenet.add(
        Dense(
            inputs=[(dense_input_shape, dense_input)],
            outputs=500,
            activation='tanh'
        )
    )
    # automatically hook a softmax classification layer, outputting the probabilities.
    lenet.add(
        Softmax, outputs=10, out_as_probs=True
    )

    return lenet


if __name__ == '__main__':
    # Grab the MNIST dataset
    data = MNIST(concat_train_valid=False)

    # we need to convert the (784,) flat example from MNIST to (1, 28, 28) for a 2D greyscale image
    process_mnist = lambda img: np.reshape(img, (1, 28, 28))

    # we can do this by using ModifyStreams over the inputs!
    data.train_inputs = ModifyStream(data.train_inputs, process_mnist)
    data.valid_inputs = ModifyStream(data.valid_inputs, process_mnist)
    data.test_inputs = ModifyStream(data.test_inputs, process_mnist)

    # now build the actual model
    lenet = build_lenet()
    # define our loss to optimize for the model (and the target variable)
    # targets from MNIST are int64 numbers 0-9
    y = lvector('y')
    loss = Neg_LL(inputs=lenet.get_outputs(), targets=y, one_hot=False)
    error_monitor = Monitor(name='error', expression=mean(neq(lenet.models[-1].y_pred, y)), valid=True, test=True)
    # optimize our model to minimize loss given the dataset using SGD
    optimizer = SGD(model=lenet,
                    dataset=data,
                    loss=loss,
                    epochs=200,
                    batch_size=500,
                    learning_rate=.1,
                    momentum=False)
    optimizer.train(monitor_channels=error_monitor)
