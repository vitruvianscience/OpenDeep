import numpy
import logging
import opendeep.log.logger as logger
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from opendeep.models.multi_layer.rnn_gsn import RNN_GSN
from opendeep.data.standard_datasets.image.mnist import MNIST
from opendeep.optimization.adadelta import AdaDelta
from opendeep.optimization.stochastic_gradient_descent import SGD
from opendeep.utils.image import tile_raster_images
from opendeep.utils.misc import closest_to_square_factors
from opendeep.monitor.monitor import Monitor
import PIL.Image as Image

log = logging.getLogger(__name__)


def run_sequence(sequence=0):
    log.info("Creating RNN-GSN for sequence %d!" % sequence)

    # grab the MNIST dataset
    mnist = MNIST(sequence_number=sequence, concat_train_valid=True)
    outdir = "outputs/rnngsn/mnist_%d/" % sequence

    rng = numpy.random.RandomState(1234)
    mrg = RandomStreams(rng.randint(2 ** 30))
    rnngsn = RNN_GSN(layers=2,
                     walkbacks=4,
                     input_size=28 * 28,
                     hidden_size=1000,
                     tied_weights=True,
                     rnn_hidden_size=100,
                     weights_init='uniform',
                     weights_interval='montreal',
                     rnn_weights_init='identity',
                     mrg=mrg,
                     outdir=outdir)
    # load pretrained rbm on mnist
    # rnngsn.load_gsn_params('outputs/trained_gsn_epoch_1000.pkl')
    # make an optimizer to train it (AdaDelta is a good default)
    optimizer = AdaDelta(model=rnngsn,
                         dataset=mnist,
                         n_epoch=200,
                         batch_size=100,
                         minimum_batch_size=2,
                         learning_rate=1e-6,
                         save_frequency=1,
                         early_stop_length=200)
    # optimizer = SGD(model=rnngsn,
    #                 dataset=mnist,
    #                 n_epoch=300,
    #                 batch_size=100,
    #                 minimum_batch_size=2,
    #                 learning_rate=.25,
    #                 lr_decay='exponential',
    #                 lr_factor=.995,
    #                 momentum=0.5,
    #                 nesterov_momentum=True,
    #                 momentum_decay=False,
    #                 save_frequency=20,
    #                 early_stop_length=100)

    crossentropy = Monitor('crossentropy', rnngsn.get_monitors()['noisy_recon_cost'], test=True)
    error = Monitor('error', rnngsn.get_monitors()['mse'], test=True)

    # perform training!
    optimizer.train(monitor_channels=[crossentropy, error])
    # use the generate function!
    log.debug("generating images...")
    generated, ut = rnngsn.generate(initial=None, n_steps=400)



    # Construct image
    image = Image.fromarray(
        tile_raster_images(
            X=generated,
            img_shape=(28, 28),
            tile_shape=(20, 20),
            tile_spacing=(1, 1)
        )
    )
    image.save(outdir + "rnngsn_mnist_generated.png")
    log.debug('saved generated.png')

    # Construct image from the weight matrix
    image = Image.fromarray(
        tile_raster_images(
            X=rnngsn.weights_list[0].get_value(borrow=True).T,
            img_shape=(28, 28),
            tile_shape=closest_to_square_factors(rnngsn.hidden_size),
            tile_spacing=(1, 1)
        )
    )
    image.save(outdir + "rnngsn_mnist_weights.png")

    log.debug("done!")

    del mnist
    del rnngsn
    del optimizer


if __name__ == '__main__':
    # set up the logging environment to display outputs (optional)
    # although this is recommended over print statements everywhere
    logger.config_root_logger()
    run_sequence(1)
    run_sequence(2)
    run_sequence(3)
    run_sequence(4)
