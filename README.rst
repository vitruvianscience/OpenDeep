.. image:: readme_images/OpenDeep_logo_name.png
   :scale: 50 %
   :alt: OpenDeep
   :align: center

=========================================
OpenDeep: A modular deep learning library
=========================================
Documentation: http://www.opendeep.org/

OpenDeep_ is a general purpose commercial and research grade deep learning library for Python built from the ground up
in Theano_ with a focus on flexibility and ease of use for both industry data scientists and cutting-edge researchers.

**This library is currently undergoing rapid development and is in its alpha stages.**

You can train and use existing deep learning models as a black box implementation, combine multiple models
to create your own novel research, or write new models from scratch without worrying about the overhead!

.. image:: readme_images/automate!.jpg
   :align: center

.. _OpenDeep: http://www.opendeep.org/
.. _Theano: http://deeplearning.net/software/theano/

Motivation
----------

- **Modularity**. A lot of recent deep learning progress has come from combining multiple models. Existing libraries are either too confusing or not easily extensible enough to perform novel research and also quickly set up existing algorithms at scale. This need for transparency and modularity is the main motivating factor for creating the OpenDeep library, where we hope novel research and industry use can both be easily implemented.

- **Ease of use**. Many libraries require a lot of familiarity with deep learning or their specific package structures. OpenDeep's goal is to be the best-documented deep learning library and have smart enough default code that someone without a background can start training models, while experienced practitioners can easily create and customize their own algorithms. OpenDeep is a 'black box' factory - it has all the parts you need to make your own 'black boxes', or you could use existing ones.

- **State of the art**. A side effect of modularity and ease of use, OpenDeep aims to maintain state-of-the-art performance as new algorithms and papers get published. As a research library, citing and accrediting those authors and code used is very important to the library.


Installation
------------
Because OpenDeep is still in alpha, you have to install via setup.py.

Dependencies
^^^^^^^^^^^^

* Theano_: Theano and its dependencies are required to use OpenDeep. You need to install the bleeding-edge version, which has `installation instructions here`_.

  * For GPU integration with Theano, you also need the latest `CUDA drivers`_. Here are `instructions for setting up Theano for the GPU`_. If you prefer to use a server on Amazon Web Services, here are instructions for setting up an `EC2 server with Theano`_.

  * CuDNN_ (optional): for a fast convolutional net support from Nvidia. You will want to move the files to Theano's directory like the instructions say here: `Theano cuDNN integration`_.

* `Pillow (PIL)`_: image manipulation functionality.

* PyYAML_ (optional): used for YAML parsing of config files.

.. _installation instructions here: http://deeplearning.net/software/theano/install.html#bleeding-edge-install-instructions

.. _CUDA drivers: https://developer.nvidia.com/cuda-toolkit
.. _instructions for setting up Theano for the GPU: http://deeplearning.net/software/theano/tutorial/using_gpu.html
.. _EC2 server with Theano: http://markus.com/install-theano-on-aws

.. _CuDNN: https://developer.nvidia.com/cuDNN
.. _Theano cuDNN integration: http://deeplearning.net/software/theano/library/sandbox/cuda/dnn.html

.. _Pillow (PIL): https://pillow.readthedocs.org/installation.html

.. _PyYAML: http://pyyaml.org/

Install from source
^^^^^^^^^^^^^^^^^^^
1) Navigate to your desired installation directory and download the github repository::

    git clone https://github.com/vitruvianscience/opendeep.git

2) Navigate to the top-level folder (should be named OpenDeep and contain the file setup.py) and run setup.py with develop mode::

    cd opendeep
    python setup.py develop

Using develop instead of the normal <python setup.py install> allows you to update the repository files by pulling
from git and have the whole package update! No need to reinstall.

That's it! Now you should be able to import opendeep into python modules.

Quick Start
-----------
To get up to speed on deep learning, check out a blog post here: `Deep Learning 101`_.
You can also go through guides on OpenDeep's documentation site: http://www.opendeep.org/

Let's say you want to train a Denoising Autoencoder on the MNIST handwritten digit dataset. You can get started
in just a few lines of code::

    # standard libraries
    import logging
    # third-party imports
    from opendeep.log.logger import config_root_logger
    import opendeep.data.dataset as datasets
    from opendeep.data.standard_datasets.image.mnist import MNIST
    from opendeep.models.single_layer.autoencoder import DenoisingAutoencoder
    from opendeep.optimization.adadelta import AdaDelta

    # grab the logger to record our progress
    log = logging.getLogger(__name__)
    # set up the logging to display to std.out and files.
    config_root_logger()
    log.info("Creating a new Denoising Autoencoder")

    # create the MNIST dataset
    mnist = MNIST()

    # define some model configuration parameters
    config = {
        "input_size": 28*28, # dimensions of the MNIST images
        "hidden_size": 1500  # number of hidden units - generally bigger than input size
    }
    # create the denoising autoencoder
    dae = DenoisingAutoencoder(config)

    # create the optimizer to train the denoising autoencoder
    # AdaDelta is normally a good generic optimizer
    optimizer = AdaDelta(dae, mnist)
    optimizer.train()

    # test the trained model and save some reconstruction images
    n_examples = 100
    # grab 100 test examples
    test_xs = mnist.getDataByIndices(indices=range(n_examples), subset=datasets.TEST)
    # test and save the images
    dae.create_reconstruction_image(test_xs)


Congrats, you just:

- set up a dataset (MNIST)

- instantiated a denoising autoencoder model with some configurations

- trained it with an AdaDelta optimizer

- and predicted some outputs given inputs (and saved them as an image)!

.. _Deep Learning 101: http://markus.com/deep-learning-101/


More Information
----------------
Source code: https://github.com/vitruvianscience/opendeep

Documentation: http://www.opendeep.org/

User group: `opendeep-users`_

Developer group: `opendeep-dev`_

We would love all help to make this the best library possible! Feel free to fork the repository and
join the Google groups!

.. _opendeep-users: https://groups.google.com/forum/#!forum/opendeep-users/
.. _opendeep-dev: https://groups.google.com/forum/#!forum/opendeep-dev/
