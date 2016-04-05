import numpy as np
import theano
from theano.tensor import tensor4, matrix, mean, neq, lvector
from opendeep import config_root_logger
from opendeep.data import ModifyStream
from opendeep.models import Prototype, Conv2D, Dense, Softmax
from opendeep.models.utils import Pool2D, Flatten
from opendeep.monitor import Monitor, FileService
from opendeep.optimization.loss import Neg_LL
from opendeep.optimization import AdaDelta
from opendeep.data import CIFAR10

# grab a log to output useful info
config_root_logger()

def build_cifar():
    # quick and dirty way to create a model from arbitrary layers
    cifar = Prototype()

    x = ((None, 3, 32, 32), tensor4('x'))

    # our first convolutional layer
    cifar.add(
        Conv2D(inputs=x, n_filters=16, filter_size=5, activation='relu')
    )
    # our first pooling layer, automatically hooking inputs to the previous convolutional outputs
    cifar.add(
        Pool2D, size=2
    )
    # our second convolutional layer
    cifar.add(
        Conv2D, n_filters=20, filter_size=5, activation='relu'
    )
    # our second pooling layer
    cifar.add(
        Pool2D, size=2
    )
    # now we need to flatten the 4D convolution outputs into 2D matrix (just flatten the trailing dimensions)
    cifar.add(
        Flatten, ndim=2
    )
    # hook a softmax classification layer, outputting the probabilities.
    cifar.add(
        Softmax, outputs=10, out_as_probs=True
    )

    return cifar


if __name__ == '__main__':
    # Grab the MNIST dataset
    data = CIFAR10()
    print data.train_targets.shape
    # now build the actual model
    cifar = build_cifar()
    # define our loss to optimize for the model (and the target variable)
    y = lvector('y')
    loss = Neg_LL(inputs=cifar.get_outputs(), targets=y, one_hot=False)
    # optimize our model to minimize loss given the dataset using SGD
    optimizer = AdaDelta(model=cifar,
                    dataset=data,
                    loss=loss,
                    epochs=200,
                    batch_size=128,
                    learning_rate=.01)
    def cb():
        # grab the activations
        activations = [layer.get_outputs() for layer in cifar]
        n_layers = len(activations)
        f_acts = theano.function(inputs=cifar.get_inputs(), outputs=activations)
        # visualize the model activations for a random example in test_inputs
        for i, x in enumerate(data.train_inputs):
            if i > 0:
                break
            in_example = np.asarray([x])
            acts = f_acts(in_example)

    optimizer.train(callback=cb)
