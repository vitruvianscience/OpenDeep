"""
Please refer to the following tutorial in the documentation at www.opendeep.org

Tutorial: Your Second Model (Combining Layers)
"""
# standard libraries
import logging
# third party libraries
from opendeep.log.logger import config_root_logger
from opendeep.models.container import Prototype
from opendeep.tutorials.tutorial01_modular_dae import DenoisingAutoencoder
from opendeep.optimization.adadelta import AdaDelta

# grab a log to output useful info
log = logging.getLogger(__name__)

def main():
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
    optimizer = AdaDelta(model=stacked_dae, )




if __name__ == '__main__':
    config_root_logger()
    main()