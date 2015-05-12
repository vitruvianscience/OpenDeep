import numpy
import logging
import opendeep.log.logger as logger
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from opendeep.models.multi_layer.rnn_rbm import RNN_RBM
from opendeep.data.standard_datasets.image.mnist import MNIST
from opendeep.optimization.adadelta import AdaDelta
from opendeep.utils.image import tile_raster_images
from opendeep.utils.misc import closest_to_square_factors
from opendeep.monitor.monitor import Monitor
from opendeep.monitor.plot import Plot
import PIL.Image as Image

log = logging.getLogger(__name__)


def run_sequence(sequence=0):
    log.info("Creating RNN-RBM for sequence %d!" % sequence)

    # grab the MNIST dataset
    mnist = MNIST(sequence_number=sequence, concat_train_valid=True)
    outdir = "outputs/rnnrbm/mnist_%d/" % sequence
    # create the RNN-RBM
    rng = numpy.random.RandomState(1234)
    mrg = RandomStreams(rng.randint(2 ** 30))
    rnnrbm = RNN_RBM(input_size=28 * 28,
                     hidden_size=1000,
                     rnn_hidden_size=100,
                     k=15,
                     weights_init='uniform',
                     weights_interval=4 * numpy.sqrt(6. / (28 * 28 + 500)),
                     rnn_weights_init='identity',
                     rnn_hidden_activation='relu',
                     rnn_weights_std=1e-4,
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
                         learning_rate=1e-8,
                         save_frequency=10,
                         early_stop_length=200)

    crossentropy = Monitor('crossentropy', rnnrbm.get_monitors()['crossentropy'], test=True)
    error = Monitor('error', rnnrbm.get_monitors()['mse'], test=True)
    plot = Plot(bokeh_doc_name='rnnrbm_mnist_%d' % sequence, monitor_channels=[crossentropy, error], open_browser=True)

    # perform training!
    optimizer.train(plot=plot)
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
    run_sequence(2)
    run_sequence(3)
    run_sequence(4)
