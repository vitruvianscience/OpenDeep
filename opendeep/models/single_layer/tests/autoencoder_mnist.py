# standard libraries
import logging
# internal imports
from opendeep.log.logger import config_root_logger
from opendeep.models.single_layer.autoencoder import DenoisingAutoencoder
from opendeep.data.standard_datasets.image.mnist import MNIST
from opendeep.optimization.adadelta import AdaDelta

log = logging.getLogger(__name__)

###############################################
# MAIN METHOD FOR RUNNING DEFAULT DAE EXAMPLE #
###############################################
def run_dae():
    ########################################
    # Initialization things with arguments #
    ########################################
    config_root_logger()
    log.info("Creating a new DAE")

    mnist = MNIST()
    config = {
        "outdir": 'outputs/dae/mnist/',
        "input_size": 28*28,
        "tied_weights": True
    }
    dae = DenoisingAutoencoder(**config)

    # # Load initial weights and biases from file
    # params_to_load = 'dae_params.pkl'
    # dae.load_params(params_to_load)

    optimizer = AdaDelta(model=dae, dataset=mnist, epochs=100)
    optimizer.train()

    # Save some reconstruction output images
    n_examples = 100
    test_xs = mnist.test_inputs[:n_examples]
    dae.create_reconstruction_image(test_xs)

    del dae, mnist


if __name__ == '__main__':
    run_dae()
