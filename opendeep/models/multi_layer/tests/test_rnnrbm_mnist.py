import numpy
import theano
import logging
import opendeep.log.logger as logger
from opendeep.models.multi_layer.rnn_rbm import RNN_RBM
from opendeep.data.standard_datasets.image.mnist import MNIST
from opendeep.optimization.adadelta import AdaDelta
from opendeep.utils.image import tile_raster_images
from opendeep.utils.misc import closest_to_square_factors
import PIL.Image as Image

log = logging.getLogger(__name__)

def run_sequence(sequence=0):
    log.info("Creating RNN-RBM for sequence %d!" % sequence)

    # grab the MNIST dataset
    mnist = MNIST(sequence_number=sequence)
    outdir = "outputs/rnnrbm/mnist_%d/" % sequence
    # create the RNN-RBM
    rng = numpy.random.RandomState(1234)
    mrg = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(2 ** 30))
    rnnrbm = RNN_RBM(input_size=28 * 28,
                     hidden_size=1000,
                     recurrent_hidden_size=100,
                     k=15,
                     weights_init='uniform',
                     weights_interval=4 * numpy.sqrt(6. / (28 * 28 + 500)),
                     recurrent_weights_init='gaussian',
                     recurrent_weights_std=1e-4,
                     rng=rng,
                     mrg=mrg,
                     outdir=outdir)
    # load pretrained rbm on mnist
    # rnnrbm.load_params(outdir + 'trained_epoch_200.pkl')
    # make an optimizer to train it (AdaDelta is a good default)
    optimizer = AdaDelta(model=rnnrbm,
                         dataset=mnist,
                         n_epoch=200,
                         batch_size=100,
                         minimum_batch_size=2,
                         learning_rate=1e-6,
                         save_frequency=10,
                         rng=rng)
    # perform training!
    optimizer.train()
    # use the generate function!
    log.debug("generating images...")
    generated, ut = rnnrbm.generate(initial=None, n_steps=400)

    # Construct image
    image = Image.fromarray(
        tile_raster_images(
            X=generated,
            img_shape=(28, 28),
            tile_shape=(20, 20),
            tile_spacing=(1, 1)
        )
    )
    image.save(outdir + "rnnrbm_mnist_generated.png")
    log.debug('saved generated.png')

    # Construct image from the weight matrix
    image = Image.fromarray(
        tile_raster_images(
            X=rnnrbm.W.get_value(borrow=True).T,
            img_shape=(28, 28),
            tile_shape=closest_to_square_factors(rnnrbm.hidden_size),
            tile_spacing=(1, 1)
        )
    )
    image.save(outdir + "rnnrbm_mnist_weights.png")

    log.debug("done!")

    del mnist
    del rnnrbm
    del optimizer


if __name__ == '__main__':
    # set up the logging environment to display outputs (optional)
    # although this is recommended over print statements everywhere
    logger.config_root_logger()
    run_sequence(1)
    # run_sequence(2)
    # run_sequence(3)
    # run_sequence(4)
