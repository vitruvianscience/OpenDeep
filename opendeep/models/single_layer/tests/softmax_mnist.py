from __future__ import print_function
from opendeep.models.single_layer.basic import SoftmaxLayer
# import the dataset and optimizer to use
from opendeep.data.standard_datasets.image.mnist import MNIST
from opendeep.optimization.adadelta import AdaDelta


if __name__ == '__main__':
    # set up the logging environment to display outputs (optional)
    # although this is recommended over print statements everywhere
    import logging
    from opendeep.log import config_root_logger
    config_root_logger()
    log = logging.getLogger(__name__)
    log.info("Creating softmax!")

    # grab the MNIST dataset
    mnist = MNIST()
    # create the softmax classifier
    s = SoftmaxLayer(input_size=28 * 28, output_size=10, out_as_probs=False)
    # make an optimizer to train it (AdaDelta is a good default)
    optimizer = AdaDelta(model=s, dataset=mnist, epochs=20)
    # perform training!
    optimizer.train()
    # test it on some images!
    test_data, test_labels = mnist.test_inputs[:25], mnist.test_targets[:25]
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
    optimizer = AdaDelta(model=s, dataset=mnist, epochs=20)
    # perform training!
    optimizer.train()
    # test it on some images!
    test_data, test_labels = mnist.test_inputs[:5], mnist.test_targets[:5]
    # use the run function!
    preds = s.run(test_data)
    print('-------')
    print(preds)
    print(test_labels.astype('int32'))
    del mnist
    del s
    del optimizer
