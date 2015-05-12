from __future__ import print_function
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
    # create the softmax classifier
    s = SoftmaxLayer(input_size=28 * 28, output_size=10, out_as_probs=False)
    # make an optimizer to train it (AdaDelta is a good default)
    optimizer = AdaDelta(model=s, dataset=mnist, n_epoch=20)
    # perform training!
    optimizer.train()
    # test it on some images!
    test_data, test_labels = mnist.getSubset(subset=TEST)
    test_data = test_data[:25].eval()
    test_labels = test_labels[:25].eval()
    # use the run function!
    preds = s.run(test_data)
    print('-------')
    print(preds)
    print(test_labels.astype('int32'))
    print()
    print()
    del mnist
    del s
    del optimizer


    log.info("Creating softmax with categorical cross-entropy!")
    # grab the MNIST dataset
    mnist = MNIST(one_hot=True)
    # create the softmax classifier
    s = SoftmaxLayer(input_size=28*28, output_size=10, cost='categorical_crossentropy', out_as_probs=True)
    # make an optimizer to train it (AdaDelta is a good default)
    optimizer = AdaDelta(model=s, dataset=mnist, n_epoch=20)
    # perform training!
    optimizer.train()
    # test it on some images!
    test_data, test_labels = mnist.getSubset(subset=TEST)
    test_data = test_data[:5].eval()
    test_labels = test_labels[:5].eval()
    # use the run function!
    preds = s.run(test_data)
    print('-------')
    print(preds)
    print(test_labels.astype('int32'))
    del mnist
    del s
    del optimizer