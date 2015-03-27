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
    config = {"output_path": '../../../../outputs/dae/mnist/'}
    dae = DenoisingAutoencoder(config=config, dataset=mnist)

    # # Load initial weights and biases from file
    # params_to_load = 'dae_params.pkl'
    # dae.load_params(params_to_load)

    optimizer = AdaDelta(dae, mnist)
    optimizer.train()

    # Save some reconstruction output images
    import opendeep.data.dataset as datasets
    n_examples = 100
    test_xs = mnist.getDataByIndices(indices=range(n_examples), subset=datasets.TEST)
    dae.create_reconstruction_image(test_xs)


if __name__ == '__main__':
    run_dae()