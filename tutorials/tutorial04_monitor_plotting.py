"""
Please refer to the following tutorial in the documentation at www.opendeep.org

Guide: Live Plotting and Monitoring
"""
# standard libraries
import logging
# third party libraries
import theano.tensor as T
from opendeep.log.logger import config_root_logger
from opendeep.models.container import Prototype
from opendeep.models.single_layer.basic import BasicLayer, SoftmaxLayer
from opendeep.optimization.stochastic_gradient_descent import SGD
from opendeep.data.standard_datasets.image.mnist import MNIST
from opendeep.monitor.plot import Plot
from opendeep.monitor.monitor import Monitor, MonitorsChannel

# grab a log to output useful info
config_root_logger()
log = logging.getLogger(__name__)

def main():
    # First, let's create a simple feedforward MLP with one hidden layer as a Prototype.
    mlp = Prototype()
    mlp.add(BasicLayer(input_size=28*28, output_size=1000, activation='rectifier', noise='dropout'))
    mlp.add(SoftmaxLayer(output_size=10))

    # Now, we get to choose what values we want to monitor, and what datasets we would like to monitor on!
    # Each Model (in our case, the Prototype), has a get_monitors method that will return a useful
    # dictionary of {string_name: monitor_theano_expression} for various computations of the model we might
    # care about. By default, this method returns an empty dictionary - it was the model creator's job to
    # include potential monitor values.
    mlp_monitors = mlp.get_monitors()
    mlp_channel = MonitorsChannel(name="error")
    for name, expression in mlp_monitors.items():
        mlp_channel.add(Monitor(name=name, expression=expression, train=True, valid=True, test=True))

    # create some monitors for statistics about the hidden and output weights!
    # let's look at the mean, variance, and standard deviation of the weights matrices.
    weights_channel = MonitorsChannel(name="weights")
    hiddens_1 = mlp[0].get_params()[0]
    hiddens1_mean = T.mean(hiddens_1)
    weights_channel.add(Monitor(name="hiddens_mean", expression=hiddens1_mean, train=True))

    hiddens_2 = mlp[1].get_params()[0]
    hiddens2_mean = T.mean(hiddens_2)
    weights_channel.add(Monitor(name="out_mean", expression=hiddens2_mean, train=True))

    # create our plot object to do live plotting!
    plot = Plot(bokeh_doc_name="Monitor Tutorial", monitor_channels=[mlp_channel, weights_channel], open_browser=True)

    # use SGD optimizer
    optimizer = SGD(model=mlp,
                    dataset=MNIST(concat_train_valid=False),
                    n_epoch=500,
                    save_frequency=100,
                    batch_size=600,
                    learning_rate=.01,
                    lr_decay=False,
                    momentum=.9,
                    nesterov_momentum=True)

    # train, with the plot!
    optimizer.train(plot=plot)


if __name__ == '__main__':
    main()