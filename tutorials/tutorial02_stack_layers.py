"""
Please refer to the following tutorial in the documentation at www.opendeep.org

Tutorial: Your Second Model (Combining Layers)
"""
# standard libraries
import logging
# third party libraries
from opendeep.log import config_root_logger
from opendeep.models import Prototype, Dense, SoftmaxLayer
from opendeep.optimization import AdaDelta
from opendeep.data import MNIST

# grab a log to output useful info
log = logging.getLogger(__name__)

def create_mlp():
    # define the model layers
    relu_layer1 = Dense(input_size=784, output_size=1000, activation='rectifier')
    relu_layer2 = Dense(inputs_hook=(1000, relu_layer1.get_outputs()), output_size=1000, activation='rectifier')
    class_layer3 = SoftmaxLayer(inputs_hook=(1000, relu_layer2.get_outputs()), output_size=10, out_as_probs=False)
    # add the layers as a Prototype
    mlp = Prototype(layers=[relu_layer1, relu_layer2, class_layer3])

    mnist = MNIST()

    optimizer = AdaDelta(model=mlp, dataset=mnist, epochs=20)
    optimizer.train()

    test_data, test_labels = mnist.test_inputs[:25], mnist.test_targets[:25]

    # use the run function!
    preds = mlp.run(test_data)
    log.info('-------')
    log.info("predicted: %s",str(preds))
    log.info("actual:    %s",str(test_labels.astype('int32')))

if __name__ == '__main__':
    config_root_logger()
    create_mlp()
