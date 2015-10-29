from __future__ import print_function
# third party libraries
from theano.tensor import matrix
from opendeep.log.logger import config_root_logger
from opendeep.models import Prototype, Dense, SoftmaxLayer
from opendeep.models.utils import Activation
from opendeep.optimization import AdaDelta
from opendeep.data import MNIST

def run_mlp():
    # # define the model layers
    # layer1 = Dense(input_size=784, output_size=1000, activation='rectifier')
    # layer2 = Dense(inputs_hook=(1000, layer1.get_outputs()), output_size=1000, activation='rectifier')
    # classlayer3 = SoftmaxLayer(inputs_hook=(1000, layer2.get_outputs()), output_size=10, out_as_probs=False)
    # # add the layers to the prototype
    # mlp = Prototype(layers=[layer1, layer2, classlayer3])

    # test the new way to automatically fill in inputs for models
    mlp = Prototype()
    x = ((None, 784), matrix("x"))
    mlp.add(Dense(inputs=x, outputs=1000, activation='rectifier', noise='dropout'))
    mlp.add(Dense(output_size=1500, activation='tanh', noise='dropout'))
    mlp.add(Softmax(output_size=10))

    mnist = MNIST()

    optimizer = AdaDelta(model=mlp, dataset=mnist, epochs=10)
    optimizer.train()

    test_data, test_labels = mnist.test_inputs, mnist.test_targets
    test_data = test_data[:25]
    test_labels = test_labels[:25]
    # use the run function!
    yhat = mlp.run(test_data)
    print('-------')
    print('Prediction: %s' % str(yhat))
    print('Actual:     %s' % str(test_labels.astype('int32')))



if __name__ == '__main__':
    config_root_logger()
    run_mlp()
