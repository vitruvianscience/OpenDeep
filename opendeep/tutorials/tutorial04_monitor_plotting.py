"""
Please refer to the following tutorial in the documentation at www.opendeep.org

Tutorial: Live Plotting and Monitoring
"""
# standard libraries
import logging
# third party libraries
from opendeep.log.logger import config_root_logger
from opendeep.models.container import Prototype
from opendeep.models.single_layer.basic import BasicLayer, SoftmaxLayer
from opendeep.optimization.adadelta import AdaDelta
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



if __name__ == '__main__':
    main()