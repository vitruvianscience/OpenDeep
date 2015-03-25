import theano.tensor as T
from opendeep.models.single_layer.basic import SoftmaxLayer
# import the dataset and optimizer to use
from opendeep.data.dataset import TEST
from opendeep.data.standard_datasets.image.mnist import MNIST
from opendeep.optimization.adadelta import AdaDelta


if __name__ == '__main__':
    # set up the logging environment to display outputs (optional)
    # although this is recommended over print statements everywhere
    import logging
    import opendeep.log.logger as logger
    logger.config_root_logger()
    log = logging.getLogger(__name__)
    log.info("Creating softmax!")

    # grab the MNIST dataset
    mnist = MNIST()
    # create your shiny new DAE
    s = SoftmaxLayer(input_size=28 * 28, output_size=10)
    # make an optimizer to train it (AdaDelta is a good default)
    optimizer = AdaDelta(model=s, dataset=mnist, n_epoch=50)
    # perform training!
    optimizer.train()
    # test it on some images!
    test_data = mnist.getDataByIndices(indices=range(25), subset=TEST)
    # use the predict function!
    preds = s.predict([test_data])
    print '-------'
    print T.argmax(preds, axis=1).eval()
    print mnist.getLabelsByIndices(indices=range(25), subset=TEST)
    print
    print
    del mnist
    del s
    del optimizer


    log.info("Creating softmax with categorical cross-entropy!")
    # grab the MNIST dataset
    mnist = MNIST(one_hot=True)
    # create your shiny new DAE
    s = SoftmaxLayer(input_size=28*28, output_size=10, cost='categorical_crossentropy')
    # make an optimizer to train it (AdaDelta is a good default)
    optimizer = AdaDelta(model=s, dataset=mnist, n_epoch=50)
    # perform training!
    optimizer.train()
    # test it on some images!
    test_data = mnist.getDataByIndices(indices=range(25), subset=TEST)
    # use the predict function!
    preds = s.predict([test_data])
    print '-------'
    print preds
    print mnist.getLabelsByIndices(indices=range(25), subset=TEST)
    del mnist
    del s
    del optimizer