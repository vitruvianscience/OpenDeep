============================================
OpenDeep: A modular machine learning library
============================================
OpenDeep is deep learning library built from the ground up in Theano with a focus on flexibility and ease of use
for both researchers and industry data scientists.

.. image:: readme_images/automate!.jpg

This library is currently undergoing rapid development and is in its alpha stages.

For support, you can join the google group here: `opendeep-users <https://groups.google.com/forum/#!forum/opendeep-users>`_.

Motivation
----------
- **Modularity**. A lot of recent deep learning progress has come from combining multiple models together (such as RNN + CNN for Andrej Karpathy and Fei-Fei Li. 
I have found existing libraries to be either too confusing or not easily extensible enough to perform novel research and also quickly set up existing algorithms at scale. 
This need for transparency and modularity is the main motivating factor for creating the OpenDeep library, where I hope novel research and industry use can both 
be easily implemented.
- **Ease of use**. Many libraries require a lot of familiarity with deep learning or their specific package structures. OpenDeep's goal is to be 
the best-documented deep learning library and have smart enough default code that someone without a background can start training models. This motivation 
will lead to a series of easy to understand tutorials for the different modules in the library.
- **State of the art**. A side effect of modularity and ease of use, OpenDeep aims to maintain state-of-the-art performance as new algorithms and papers 
get published. As a research library, citing and accrediting those authors and code used is very important to the library.


Installation
------------
Because OpenDeep is still in alpha, you have to install via setup.py.

First, install the dependencies.
- Theano: Theano and its dependencies are required to use OpenDeep. You need the bleeding-edge version as specified here: `Theano bleeding-edge <http://deeplearning.net/software/theano/install.html#bleeding-edge-install-instructions>`_
 I also recommend downloading CUDA to work on an Nvidia GPU, because using the GPU is orders of magnitude faster. You can find instructions for installing Theano on an 
Amazon Web Services GPU machine here: `Installing Theano on AWS for Deep Learning <http://markus.com/install-theano-on-aws/>`_ Another thing to keep in mind is using a good BLAS linked with Numpy, as that is normally a bottleneck.
- PIL: image functionality
- PyYAML (optional): used for YAML parsing
- CuDNN (optional): for a fairly fast convolutional net support from Nvidia, download the cuDNN library here: `cuDNN <https://developer.nvidia.com/cuDNN>`_ You will want to move the files to 
Theano's directory like the instructions say here: `Theano cuDNN integration <http://deeplearning.net/software/theano/library/sandbox/cuda/dnn.html>`_

Finally, to install OpenDeep, download the github repository and navigate to the top level directory. Then run:
    python setup.py develop
    
Using develop instead of the normal install allows you to update the repository files and have the whole package update!


Quick Start
-----------
To get up to speed on deep learning, check out a blog post here: `Deep Learning 101 <http://markus.com/deep-learning-101/>`_

For a view of the modularity, check out opendeep.models.model. There, you will find the base Model class, which accounts for everything 
from your basic single-layer constructs like a Softmax classification layer to your complex, multi-layer models like AlexNet. This base structure lets 
you hook models and layers to each other through their inputs, hidden representations, outputs, and parameters.

To run an example, go to the denoising autoencoder class DAE in opendeep.models.single_layer.autoencoder. There is a main() method in that file which will 
set up the logging environment, load the MNIST handwritten digit dataset, and train a denoising autoencoder on the data. You can do a keyboardInterrupt whenever you 
want to stop training early, and an image of the autoencoder's output will appear in outputs/dae/reconstruction.png. Congrats, you set up a dataset, 
instantiated a denoising autoencoder, trained it with an AdaDelta optimizer, and predicted some outputs given inputs!


Contact and Contribute
----------------------
We would love all help to make this the best library possible! Feel free to fork the repository and 
join the google group here: `opendeep-dev <https://groups.google.com/forum/#!forum/opendeep-dev/>`_


Yay you made it! Here, have some cake.

.. image:: readme_images/cake.jpg