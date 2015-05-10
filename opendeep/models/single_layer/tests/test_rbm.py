import numpy
import theano
from opendeep.models.single_layer.restricted_boltzmann_machine import RBM
from opendeep.monitor.monitor import Monitor
from opendeep.data.dataset import TEST
from opendeep.data.standard_datasets.image.mnist import MNIST
from opendeep.optimization.adadelta import AdaDelta
from opendeep.optimization.optimizer import Optimizer
from opendeep.optimization.stochastic_gradient_descent import SGD
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
    log.info("Creating RBM!")

    # grab the MNIST dataset
    mnist = MNIST(concat_train_valid=False)
    # create the RBM
    rng = numpy.random.RandomState(1234)
    mrg = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(2**30))
    rbm = RBM(input_size=28*28, hidden_size=500, k=15, weights_init='uniform', weights_interval=4*numpy.sqrt(6./(28*28+500)), mrg=mrg)
    # rbm.load_params('rbm_trained.pkl')
    # make an optimizer to train it (AdaDelta is a good default)

    # optimizer = SGD(model=rbm, dataset=mnist, batch_size=20, learning_rate=0.1, lr_decay=False, nesterov_momentum=False, momentum=False)

    optimizer = Optimizer(lr_decay=False, learning_rate=0.1, model=rbm, dataset=mnist, batch_size=20, save_frequency=1)

    ll = Monitor('pseudo-log', rbm.get_monitors()['pseudo-log'])

    # perform training!
    optimizer.train(monitor_channels=ll)
    # test it on some images!
    test_data = mnist.getSubset(TEST)[0]
    test_data = test_data[:25].eval()
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
            tile_shape=closest_to_square_factors(rbm.hidden_size),
            tile_spacing=(1, 1)
        )
    )
    image.save('rbm_weights.png')


    del mnist
    del rbm
    del optimizer