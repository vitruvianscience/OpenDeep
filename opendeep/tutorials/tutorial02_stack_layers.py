"""
Please refer to the following tutorial in the documentation at www.opendeep.org

Tutorial: Your Second Model (Combining Layers)
"""
# standard libraries
import logging
# third party libraries
from opendeep.log.logger import config_root_logger
from opendeep.models.container import Prototype
from opendeep.models.single_layer.basic import BasicLayer, SoftmaxLayer
from opendeep.tutorials.tutorial01_modular_dae import DenoisingAutoencoder
from opendeep.optimization.adadelta import AdaDelta
from opendeep.data.standard_datasets.image.mnist import MNIST
from opendeep.data.dataset import TEST

# grab a log to output useful info
log = logging.getLogger(__name__)

def stacked_dae():
    # initialize the empty container
    stacked_dae = Prototype()
    # add a few layers of denoising autoencoders!
    stacked_dae.add(DenoisingAutoencoder(input_size=28*28, hidden_size=1000))
    # use our inputs_hook we implemented before! in this case, hook the input to the output of the last model added.
    stacked_dae.add(DenoisingAutoencoder(inputs_hook=(1000, stacked_dae[-1].get_outputs()), hidden_size=1500))
    # do it again for good measure, to make a 3-layer stacked denoising autoencoder
    stacked_dae.add(DenoisingAutoencoder(inputs_hook=(1500, stacked_dae[-1].get_outputs()), hidden_size=1500))

    # that's it! we just used a container to combine three denoising autoencoders into one!
    # while this is easy to set up for experiments, I would still recommend making a Model instance
    # for your awesome new model :)

    # Let's train this container with AdaDelta on MNIST, as usual with these tutorials
    optimizer = AdaDelta(model=stacked_dae, dataset=MNIST())

    optimizer.train()

def mlp():
    # init empty container
    mlp = Prototype()
    # define the model layers
    layer1 = BasicLayer(input_size=28*28, output_size=1000, activation='rectifier')
    layer2 = BasicLayer(inputs_hook=(1000, layer1.get_outputs()), output_size=1000, activation='rectifier')
    classlayer3 = SoftmaxLayer(inputs_hook=(1000, layer2.get_outputs()), output_size=10, out_as_probs=False)

    mlp.add([layer1, layer2, classlayer3])

    mnist = MNIST()

    optimizer = AdaDelta(model=mlp, dataset=mnist)
    optimizer.train()

    test_data = mnist.getDataByIndices(indices=range(25), subset=TEST)
    # use the predict function!
    preds = mlp.predict(test_data)
    print '-------'
    print T.argmax(preds, axis=1).eval()
    print mnist.getLabelsByIndices(indices=range(25), subset=TEST)




if __name__ == '__main__':
    config_root_logger()
    mlp()
    stacked_dae()