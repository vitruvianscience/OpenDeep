"""
This module provides the main Convolutional Neural Network multi-layer models.

.. note::

    To use CuDNN wrapping, you must install the appropriate .h and .so files for theano as described here:
    http://deeplearning.net/software/theano/library/sandbox/cuda/dnn.html

"""
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
import time
# third party libraries
import theano
import theano.tensor as T
from theano.compat.python2x import OrderedDict
# internal references
from opendeep import function
from opendeep.models.model import Model
from opendeep.models.single_layer.convolutional import ConvPoolLayer
from opendeep.models.single_layer.basic import BasicLayer, SoftmaxLayer
from opendeep.utils.decorators import inherit_docs
from opendeep.utils.nnet import mirror_images
from opendeep.utils.misc import make_time_units_string, raise_to_list

log = logging.getLogger(__name__)

# Some convolution operations only work on the GPU, so do a check here:
if not theano.config.device.startswith('gpu'):
    log.warning("You should reeeeeaaaally consider using a GPU, unless this is a small toy algorithm for fun. "
                "Please enable the GPU in Theano via these instructions: "
                "http://deeplearning.net/software/theano/tutorial/using_gpu.html")

# To use the fastest convolutions possible, need to set the Theano flag as described here:
# http://benanne.github.io/2014/12/09/theano-metaopt.html
# make it THEANO_FLAGS=optimizer_including=conv_meta,metaopt.verbose=1
# OR you could set the .theanorc file with [global]optimizer_including=conv_meta [metaopt]verbose=1
if theano.config.optimizer_including != "conv_meta":
    log.warning("Theano flag optimizer_including is not conv_meta (found %s)! "
                "To have Theano cherry-pick the best convolution implementation, please set "
                "optimizer_including=conv_meta either in THEANO_FLAGS or in the .theanorc file!"
                % str(theano.config.optimizer_including))


@inherit_docs
class AlexNet(Model):
    """
    This is the base model for AlexNet, Alex Krizhevsky's efficient deep convolutional net described in:
    'ImageNet Classification with Deep Convolutional Neural Networks'
    Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
    http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf

    Most of the code here is adapted from the authors for the paper:
    'Theano-based large-scale visual recognition with multiple GPUs'
    Weiguang Ding & Ruoyan Wnag, Fei Mao, Graham Taylor
    http://arxiv.org/pdf/1412.2302.pdf
    """
    def __init__(self, inputs_hook=None, params_hook=None, outdir='outputs/alexnet/',
                 use_data_layer=False, rand_crop=True, batch_size=256):
        """
        Initialize AlexNet

        Parameters
        ----------
        inputs_hook : Tuple of (shape, variable)
            Routing information for the model to accept inputs from elsewhere. This is used for linking
            different models together. For now, you need to include the shape information (normally the
            dimensionality of the input i.e. input_size).
        params_hook : List(theano shared variable)
            A list of model parameters (shared theano variables) that you should use when constructing
            this model (instead of initializing your own shared variables). This parameter is useful when you want to
            have two versions of the model that use the same parameters - such as a training model with dropout applied
            to layers and one without for testing, where the parameters are shared between the two.
        outdir : str
            The directory you want outputs (parameters, images, etc.) to save to. If None, nothing will
            be saved.
        use_data_layer : bool
            Whether or not to mirror the input images before feeding them into the network.
        rand_crop : bool
            If `use_data_layer`, whether to randomize.
        batch_size : int
            The batch size (number of images) to process in parallel.
        """
        # combine everything by passing to Model's init
        super(AlexNet, self).__init__(**{arg: val for (arg, val) in locals().iteritems() if arg is not 'self'})
        # configs can now be accessed through self dictionary

        if self.inputs_hook or self.params_hook:
            log.error("Inputs_hook and params_hook not implemented yet for AlexNet!")

        self.flag_datalayer = use_data_layer
        self.rand_crop = rand_crop

        ####################
        # Theano variables #
        ####################
        # allocate symbolic variables for the data
        # 'rand' is a random array used for random cropping/mirroring of data
        self.x = T.tensor4('x')
        self.rand = T.vector('rand')

        ##########
        # params #
        ##########
        self.params = []
        self.noise_switches = []

        # make the network!
        self._build_computation_graph()

    def _build_computation_graph(self):
        ###################### BUILD NETWORK ##########################
        # whether or not to mirror the input images before feeding them into the network
        if self.flag_datalayer:
            layer_1_input = mirror_images(input=self.x,
                                          image_shape=(self.batch_size, 3, 256, 256),  # bc01 format
                                          cropsize=227,
                                          rand=self.rand,
                                          flag_rand=self.rand_crop)
        else:
            layer_1_input = self.x  # 4D tensor (going to be in bc01 format)

        # Start with 5 convolutional pooling layers
        log.debug("convpool layer 1...")
        convpool_layer1 = ConvPoolLayer(inputs_hook=((self.batch_size, 3, 227, 227), layer_1_input),
                                        filter_shape=(96, 3, 11, 11),
                                        convstride=4,
                                        padsize=0,
                                        group=1,
                                        poolsize=3,
                                        poolstride=2,
                                        bias_init=0.0,
                                        local_response_normalization=True)
        # Add this layer's parameters!
        self.params += convpool_layer1.get_params()

        log.debug("convpool layer 2...")
        convpool_layer2 = ConvPoolLayer(inputs_hook=((self.batch_size, 96, 27, 27, ), convpool_layer1.get_outputs()),
                                        filter_shape=(256, 96, 5, 5),
                                        convstride=1,
                                        padsize=2,
                                        group=2,
                                        poolsize=3,
                                        poolstride=2,
                                        bias_init=0.1,
                                        local_response_normalization=True)
        # Add this layer's parameters!
        self.params += convpool_layer2.get_params()

        log.debug("convpool layer 3...")
        convpool_layer3 = ConvPoolLayer(inputs_hook=((self.batch_size, 256, 13, 13), convpool_layer2.get_outputs()),
                                        filter_shape=(384, 256, 3, 3),
                                        convstride=1,
                                        padsize=1,
                                        group=1,
                                        poolsize=1,
                                        poolstride=0,
                                        bias_init=0.0,
                                        local_response_normalization=False)
        # Add this layer's parameters!
        self.params += convpool_layer3.get_params()

        log.debug("convpool layer 4...")
        convpool_layer4 = ConvPoolLayer(inputs_hook=((self.batch_size, 384, 13, 13), convpool_layer3.get_outputs()),
                                        filter_shape=(384, 384, 3, 3),
                                        convstride=1,
                                        padsize=1,
                                        group=2,
                                        poolsize=1,
                                        poolstride=0,
                                        bias_init=0.1,
                                        local_response_normalization=False)
        # Add this layer's parameters!
        self.params += convpool_layer4.get_params()

        log.debug("convpool layer 5...")
        convpool_layer5 = ConvPoolLayer(inputs_hook=((self.batch_size, 384, 13, 13), convpool_layer4.get_outputs()),
                                        filter_shape=(256, 384, 3, 3),
                                        convstride=1,
                                        padsize=1,
                                        group=2,
                                        poolsize=3,
                                        poolstride=2,
                                        bias_init=0.0,
                                        local_response_normalization=False)
        # Add this layer's parameters!
        self.params += convpool_layer5.get_params()

        # Now onto the fully-connected layers!
        fc_config = {
            'activation': 'rectifier',  # type of activation function to use for output
            'weights_init': 'gaussian',  # either 'gaussian' or 'uniform' - how to initialize weights
            'weights_mean': 0.0,  # mean for gaussian weights init
            'weights_std': 0.005,  # standard deviation for gaussian weights init
            'bias_init': 0.0  # how to initialize the bias parameter
        }
        log.debug("fully connected layer 1 (model layer 6)...")
        # we want to have dropout applied to the training version, but not the test version.
        fc_layer6_input = T.flatten(convpool_layer5.get_outputs(), 2)
        fc_layer6 = BasicLayer(inputs_hook=(9216, fc_layer6_input),
                               output_size=4096,
                               noise='dropout',
                               noise_level=0.5,
                               **fc_config)
        # Add this layer's parameters!
        self.params += fc_layer6.get_params()
        # Add the dropout noise switch
        self.noise_switches += fc_layer6.get_noise_switch()

        log.debug("fully connected layer 2 (model layer 7)...")
        fc_layer7 = BasicLayer(inputs_hook=(4096, fc_layer6.get_outputs()),
                               output_size=4096,
                               noise='dropout',
                               noise_level=0.5,
                               **fc_config)

        # Add this layer's parameters!
        self.params += fc_layer7.get_params()
        # Add the dropout noise switch
        self.noise_switches += fc_layer7.get_noise_switch()

        # last layer is a softmax prediction output layer
        softmax_config = {
            'weights_init': 'gaussian',
            'weights_mean': 0.0,
            'weights_std': 0.005,
            'bias_init': 0.0
        }
        log.debug("softmax classification layer (model layer 8)...")
        softmax_layer8 = SoftmaxLayer(inputs_hook=(4096, fc_layer7.get_outputs()),
                                      output_size=1000,
                                      **softmax_config)

        # Add this layer's parameters!
        self.params += softmax_layer8.get_params()

        # finally the softmax output from the whole thing!
        self.output = softmax_layer8.get_outputs()
        self.targets = softmax_layer8.get_targets()

        #####################
        # Cost and monitors #
        #####################
        self.train_cost = softmax_layer8.negative_log_likelihood()
        cost = softmax_layer8.negative_log_likelihood()
        errors = softmax_layer8.errors()
        train_errors = softmax_layer8.errors()

        self.monitors = OrderedDict([('cost', cost), ('errors', errors), ('dropout_errors', train_errors)])

        #########################
        # Compile the functions #
        #########################
        log.debug("Compiling functions!")
        t = time.time()
        log.debug("f_run...")
        # use the actual argmax from the classification
        self.f_run = function(inputs=[self.x], outputs=softmax_layer8.get_argmax_prediction())
        log.debug("compilation took %s", make_time_units_string(time.time() - t))

    def get_inputs(self):
        return [self.x]

    def get_outputs(self):
        return self.output

    def get_targets(self):
        return self.targets

    def get_noise_switch(self):
        return self.get_noise_switch()

    def get_train_cost(self):
        return self.train_cost

    def get_monitors(self):
        return self.monitors

    def get_params(self):
        return self.params

    def run(self, input):
        # because we use the splat to account for multiple inputs to the function, make sure input is a list.
        input = raise_to_list(input)
        # return the results of the run function!
        output = self.f_run(*input)
        return output