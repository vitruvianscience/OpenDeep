import numpy
import theano
from opendeep.models.multi_layer.rnn_rbm import RNN_RBM
from opendeep.data.standard_datasets.image.mnist import MNIST
from opendeep.optimization.adadelta import AdaDelta
# from opendeep.optimization.stochastic_gradient_descent import SGD
from opendeep.utils.image import tile_raster_images
from opendeep.utils.misc import closest_to_square_factors
import PIL.Image as Image


if __name__ == '__main__':
    # set up the logging environment to display outputs (optional)
    # although this is recommended over print statements everywhere
    import logging
    import opendeep.log.logger as logger
    logger.config_root_logger()
    log = logging.getLogger(__name__)
    log.info("Creating RNN-RBM!")

    # grab the MNIST dataset
    mnist = MNIST()
    # put the images in sequence
    mnist.sequence(sequence_number=1)
    # create the RNN-RBM
    rng = numpy.random.RandomState(1234)
    mrg = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(2**30))
    rnnrbm = RNN_RBM(input_size=28*28,
                     hidden_size=1000,
                     recurrent_hidden_size=100,
                     k=15,
                     weights_init='uniform',
                     weights_interval=4*numpy.sqrt(6./(28*28+500)),
                     recurrent_weights_init='gaussian',
                     recurrent_weights_std=1e-4,
                     rng=rng)
    # load pretrained rbm on mnist
    # rnnrbm.load_rbm_params('rbm_trained.pkl')
    # make an optimizer to train it (AdaDelta is a good default)
    # optimizer = SGD(model=rbm, dataset=mnist, n_epoch=20, batch_size=100, learning_rate=0.1, lr_decay='exponential', lr_factor=1, nesterov_momentum=False)
    optimizer = AdaDelta(model=rnnrbm, dataset=mnist, n_epoch=200, batch_size=100, learning_rate=1e-6, save_frequency=1)
    # perform training!
    optimizer.train()
    # use the generate function!
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
    image.save('generated.png')
    print 'saved generated.png'

    # Construct image from the weight matrix
    image = Image.fromarray(
        tile_raster_images(
            X=rnnrbm.W.get_value(borrow=True).T,
            img_shape=(28, 28),
            tile_shape=closest_to_square_factors(rnnrbm.hidden_size),
            tile_spacing=(1, 1)
        )
    )
    image.save('weights.png')

    print "done!"
    del mnist
    del rnnrbm
    del optimizer