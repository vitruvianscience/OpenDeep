from __future__ import print_function
# standard libraries
import logging
# third party libraries
from opendeep.log.logger import config_root_logger
from opendeep.models.container import Prototype
from opendeep.models.single_layer.basic import Dense, SoftmaxLayer
from opendeep.optimization.adadelta import AdaDelta
from opendeep.data.standard_datasets.image.mnist import MNIST

# grab a log to output useful info
log = logging.getLogger(__name__)

def run_stacked_dae():
    # stacked_dae = Prototype()
    # stacked_dae.add(DenoisingAutoencoder(input_size=784, hidden_size=1000))
    # stacked_dae.add(DenoisingAutoencoder(inputs_hook=(1000, stacked_dae[-1].get_hiddens()), hidden_size=1500))
    # stacked_dae.add(DenoisingAutoencoder(inputs_hook=(1500, stacked_dae[-1].get_hiddens()), hidden_size=1500))
    # # now we need to go back down to the input level - use params_hook to tie parameters!
    # stacked_dae.add(DenoisingAutoencoder(hiddens_hook=(1500, stacked_dae[-1].get_outputs()), input_size=1500,
    #                                      params_hook=stacked_dae[-1].get_params()))
    # stacked_dae.add(DenoisingAutoencoder(hiddens_hook=(1500, stacked_dae[-1].get_outputs()), input_size=1000,
    #                                      params_hook=stacked_dae[-1].get_params()))
    # stacked_dae.add(DenoisingAutoencoder(hiddens_hook=(100, stacked_dae[-1].get_outputs()), input_size=28 * 28,
    #                                      params_hook=stacked_dae[-1].get_params()))
    #
    # mnist = MNIST()
    #
    # optimizer = AdaDelta(model=stacked_dae, dataset=MNIST(), epochs=20)
    # optimizer.train()
    #
    pass

def run_mlp():
    # # define the model layers
    # layer1 = Dense(input_size=784, output_size=1000, activation='rectifier')
    # layer2 = Dense(inputs_hook=(1000, layer1.get_outputs()), output_size=1000, activation='rectifier')
    # classlayer3 = SoftmaxLayer(inputs_hook=(1000, layer2.get_outputs()), output_size=10, out_as_probs=False)
    # # add the layers to the prototype
    # mlp = Prototype(layers=[layer1, layer2, classlayer3])

    # test the new way to automatically fill in inputs_hook for models
    mlp = Prototype()
    mlp.add(Dense(input_size=784, output_size=1000, activation='rectifier', noise='dropout'))
    mlp.add(Dense(output_size=1500, activation='tanh', noise='dropout'))
    mlp.add(SoftmaxLayer(output_size=10))

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
    run_stacked_dae()
