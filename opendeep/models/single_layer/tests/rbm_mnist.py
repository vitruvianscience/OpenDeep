import numpy
import theano
from opendeep.models import RBM
from opendeep.monitor import Monitor
from opendeep.data import MNIST
from opendeep.optimization import Optimizer
from opendeep.utils.image import tile_raster_images
from opendeep.utils.misc import closest_to_square_factors
try:
    import PIL.Image as Image
except ImportError:
    import Image


if __name__ == '__main__':
    # set up the logging environment to display outputs (optional)
    # although this is recommended over print statements everywhere
    import logging
    from opendeep import config_root_logger
    config_root_logger()
    log = logging.getLogger(__name__)
    log.info("Creating RBM!")

    # grab the MNIST dataset
    mnist = MNIST(concat_train_valid=False)
    # create the RBM
    rng = numpy.random.RandomState(1234)
    mrg = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(2**30))
    config_args = {
        'inputs': (28*28, theano.tensor.matrix('x')),
        'hiddens': 500,
        'k': 15,
        'weights_init': 'uniform',
        'weights_interval': 4*numpy.sqrt(6./28*28+500),
        'mrg': mrg
    }
    rbm = RBM(**config_args)
    # rbm.load_params('outputs/rbm/trained_epoch_15.pkl')

    optimizer = Optimizer(learning_rate=0.1, model=rbm, dataset=mnist, batch_size=20, epochs=15)


    # perform training!
    optimizer.train()
    # test it on some images!
    test_data = mnist.test_inputs[:25]
    # use the run function!
    preds = rbm.run(test_data)

    # Construct image from the test matrix
    image = Image.fromarray(
        tile_raster_images(
            X=test_data,
            img_shape=(28, 28),
            tile_shape=(5, 5),
            tile_spacing=(1, 1)
        )
    )
    image.save('rbm_test.png')

    # Construct image from the preds matrix
    image = Image.fromarray(
        tile_raster_images(
            X=preds,
            img_shape=(28, 28),
            tile_shape=(5, 5),
            tile_spacing=(1, 1)
        )
    )
    image.save('rbm_preds.png')

    # Construct image from the weight matrix
    image = Image.fromarray(
        tile_raster_images(
            X=rbm.W.get_value(borrow=True).T,
            img_shape=(28, 28),
            tile_shape=closest_to_square_factors(config_args['hiddens']),
            tile_spacing=(1, 1)
        )
    )
    image.save('rbm_weights.png')

    del mnist
    del rbm
    del optimizer
