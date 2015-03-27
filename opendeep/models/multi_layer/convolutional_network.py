"""
.. module:: convolutional_network

This module provides the main Convolutional Neural Network multi-layer models.

TO USE CUDNN WRAPPING, YOU MUST INSTALL THE APPROPRIATE .h and .so FILES FOR THEANO LIKE SO:
http://deeplearning.net/software/theano/library/sandbox/cuda/dnn.html
"""
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Weiguang Ding", "Ruoyan Wang", "Fei Mao", "Graham Taylor", "Markus Beissinger"]
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
from opendeep.utils.nnet import mirror_images
from opendeep.utils.noise import dropout
from opendeep.utils.misc import make_time_units_string

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

class AlexNet(Model):
    """
    This is the base model for AlexNet, Alex Krizhevsky's efficient deep convolutional net described in:
    'ImageNet Classification with Deep Convolutional Neural Networks'
    Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
    http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf

    Most of the code here is adapted from the authors listed in the license above, from the paper:
    'Theano-based large-scale visual recognition with multiple GPUs'
    Weiguang Ding & Ruoyan Wnag, Fei Mao, Graham Taylor
    http://arxiv.org/pdf/1412.2302.pdf

    Copyright (c) 2014, Weiguang Ding, Ruoyan Wang, Fei Mao and Graham Taylor
    All rights reserved.
    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
        1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
        2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
        3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    defaults = {  # data stuff
                  "use_data_layer": False,
                  "rand_crop": True,
                  "batch_size": 256,  # convolutional nets are particular about the batch size
                  "output_path": '/outputs/alexnet/'
    }
    def __init__(self, config=None, defaults=defaults, inputs_hook=None, hiddens_hook=None, params_hook=None,
                 use_data_layer=None, rand_crop=None, batch_size=None):
        # combine everything by passing to Model's init
        super(AlexNet, self).__init__(**{arg: val for (arg, val) in locals().iteritems() if arg is not 'self'})
        # configs can now be accessed through self dictionary

        if self.inputs_hook or self.hiddens_hook or self.params_hook:
            log.error("Inputs_hook, hiddens_hook, and params_hook not implemented yet for AlexNet!")

        self.flag_datalayer = self.use_data_layer

        ####################
        # Theano variables #
        ####################
        # allocate symbolic variables for the data
        # 'rand' is a random array used for random cropping/mirroring of data
        self.x = T.ftensor4('x')
        self.y = T.lvector('y')
        self.rand = T.fvector('rand')

        ##########
        # params #
        ##########
        self.params = []

        # make the network!
        self.build_computation_graph()

    def build_computation_graph(self):
        ###################### BUILD NETWORK ##########################
        # whether or not to mirror the input images before feeding them into the network
        if self.flag_datalayer:
            layer_1_input = mirror_images(input=self.x,
                                          image_shape=(self.batch_size, 3, 256, 256),  # bc01 format
                                          cropsize=227,
                                          rand=self.rand,
                                          flag_rand=self.rand_crop)
        else:
            layer_1_input = self.x  # 4D tensor (going to be in c01b format)

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
        fc_layer6 = BasicLayer(inputs_hook=(9216, fc_layer6_input), output_size=4096, config=fc_config)
        # Add this layer's parameters!
        self.params += fc_layer6.get_params()

        # now apply dropout to the output for training
        dropout_layer6 = dropout(fc_layer6.get_outputs(), corruption_level=0.5)

        log.debug("fully connected layer 2 (model layer 7)...")
        fc_layer7       = BasicLayer(inputs_hook=(4096, fc_layer6.get_outputs()),
                                     output_size=4096,
                                     config=fc_config)
        fc_layer7_train = BasicLayer(inputs_hook=(4096, dropout_layer6),
                                     output_size=4096,
                                     params_hook=fc_layer7.get_params(),
                                     config=fc_config)
        # Add this layer's parameters!
        self.params += fc_layer7.get_params()

        # apply dropout again for training
        dropout_layer7 = dropout(fc_layer7_train.get_outputs(), corruption_level=0.5)

        # last layer is a softmax prediction output layer
        softmax_config = {
            'weights_init': 'gaussian',
            'weights_mean': 0.0,
            'weights_std': 0.005,
            'bias_init': 0.0
        }
        log.debug("softmax classification layer (model layer 8)...")
        softmax_layer8       = SoftmaxLayer(inputs_hook=(4096, fc_layer7.get_outputs()),
                                            output_size=1000,
                                            config=softmax_config)
        softmax_layer8_train = SoftmaxLayer(inputs_hook=(4096, dropout_layer7),
                                            output_size=1000,
                                            params_hook=softmax_layer8.get_params(),
                                            config=softmax_config)
        # Add this layer's parameters!
        self.params += softmax_layer8.get_params()

        # finally the softmax output from the whole thing!
        self.output = softmax_layer8.get_outputs()

        #####################
        # Cost and monitors #
        #####################
        self.train_cost = softmax_layer8_train.negative_log_likelihood(self.y)
        cost = softmax_layer8.negative_log_likelihood(self.y)
        errors = softmax_layer8.errors(self.y)
        train_errors = softmax_layer8_train.errors(self.y)

        self.monitors = OrderedDict([('cost', cost), ('errors', errors), ('dropout_errors', train_errors)])

        #########################
        # Compile the functions #
        #########################
        log.debug("Compiling functions!")
        t = time.time()
        log.debug("f_predict...")
        # use the actual argmax from the classification
        self.f_predict = function(inputs=[self.x], outputs=softmax_layer8.get_argmax_prediction())
        log.debug("compilation took %s", make_time_units_string(time.time() - t))

    def get_inputs(self):
        """
        This should return the input(s) to the model's computation graph. This is called by the Optimizer when creating
        the theano train function on the cost expression returned by get_train_cost().

        This should normally return the same theano variable list that is used in the inputs= argument to the f_predict
        function.
        ------------------

        :return: Theano variables representing the input(s) to the training function.
        :rtype: List(theano variable)
        """
        return [self.x]

    def get_outputs(self):
        """
        This method will return the model's output variable expression from the computational graph.
        This should be what is given for the outputs= part of the 'f_predict' function from self.predict().

        This will be used for creating hooks to link models together, where these outputs can be strung as the inputs
        or hiddens to another model :)
        ------------------

        :return: theano expression of the outputs from this model's computation
        :rtype: theano tensor (expression)
        """
        return self.output

    def get_train_cost(self):
        """
        This returns the expression that represents the cost given an input, which is used for the Optimizer during
        training. The reason we can't just compile a f_train theano function is because updates need to be calculated
        for the parameters during gradient descent - and these updates are created in the Optimizer object.
        ------------------

        :return: theano expression of the model's training cost, from which parameter gradients will be computed.
        :rtype: theano tensor
        """
        return self.train_cost

    def get_monitors(self):
        """
        This returns a dictionary of (monitor_name: monitor_expression) of variables (monitors) whose values we care
        about during training. For every monitor returned by this method, the function will be run on the
        train/validation/test dataset and its value will be reported.
        ------------------

        :return: Dictionary of String: theano_function for each monitor variable we care about in the model.
        :rtype: Dictionary
        """
        return self.monitors

    def get_params(self):
        """
        This returns the list of theano shared variables that will be trained by the Optimizer.
        These parameters are used in the gradient.
        ------------------

        :return: flattened list of theano shared variables to be trained
        :rtype: List(shared_variables)
        """
        return self.params