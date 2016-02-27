.. image:: https://readthedocs.org/projects/opendeep/badge/?version=latest
    :target: https://readthedocs.org/projects/opendeep/?badge=latest
    :alt: Documentation Status


.. image:: readme_images/OpenDeep_logo_name.png
   :scale: 50 %
   :alt: OpenDeep
   :align: center

========================================================================
OpenDeep: a fully modular & extensible deep learning framework in Python
========================================================================
Developer hub: http://www.opendeep.org/

OpenDeep_ is a deep learning framework for Python built from the ground up
in Theano_ with a focus on flexibility and ease of use for both industry data scientists and cutting-edge researchers.
OpenDeep is a modular and easily extensible framework for constructing any neural network architecture to
solve your problem.

Use OpenDeep to:

* Quickly prototype complex networks through a focus on complete modularity and containers similar to Torch.
* Configure and train existing state-of-the-art models.
* Write your own models from scratch in Theano and plug into OpenDeep for easy training and dataset integration.
* Use visualization and debugging tools to see exactly what is happening with your neural net architecture.
* Plug into your existing Numpy/Scipy/Pandas/Scikit-learn pipeline.
* Run on the CPU or GPU.

**This library is currently undergoing rapid development and is in its alpha stages.**

.. _OpenDeep: http://www.opendeep.org/
.. _Theano: http://deeplearning.net/software/theano/


Quick example usage
===================
Train and evaluate a Multilayer Perceptron (MLP - your generic feedforward neural network for classification)
on the MNIST handwritten digit dataset::

    from opendeep.models import Prototype, Dense, Softmax
    from opendeep.models.utils import Noise
    from opendeep.optimization.loss import Neg_LL
    from opendeep.optimization import AdaDelta
    from opendeep.data import MNIST
    from theano.tensor import matrix, lvector

    print "Getting data..."
    data = MNIST()

    print "Creating model..."
    in_shape = (None, 28*28)
    in_var = matrix('xs')
    mlp = Prototype()
    mlp.add(Dense(inputs=(in_shape, in_var), outputs=512, activation='relu'))
    mlp.add(Noise, noise='dropout', noise_level=0.5)
    mlp.add(Dense, outputs=512, activation='relu')
    mlp.add(Noise, noise='dropout', noise_level=0.5)
    mlp.add(Softmax, outputs=10, out_as_probs=False)

    print "Training..."
    target_var = lvector('ys')
    loss = Neg_LL(inputs=mlp.models[-1].p_y_given_x, targets=target_var, one_hot=False)

    optimizer = AdaDelta(model=mlp, loss=loss, dataset=data, epochs=10)
    optimizer.train()

    print "Predicting..."
    predictions = mlp.run(data.test_inputs)

    print "Accuracy: ", float(sum(predictions==data.test_targets)) / len(data.test_targets)

Congrats, you just:

- set up a dataset (MNIST)

- instantiated a Prototype container model

- added fully-connected (dense) layers and dropout noise to create an MLP

- trained it with an AdaDelta optimizer

- and predicted some outputs given inputs!

.. image:: readme_images/gatsby.gif
   :scale: 100 %
   :alt: Working example!
   :align: center


Installation
============
Because OpenDeep is still in alpha, you have to install via setup.py. Also, please make sure you have these dependencies installed first.

Dependencies
------------
* Theano_: Theano and its dependencies are required to use OpenDeep. You need to install the bleeding-edge version directly from their GitHub, which has `installation instructions here`_.

  * For GPU integration with Theano, you also need the latest `CUDA drivers`_. Here are `instructions for setting up Theano for the GPU`_. If you prefer to use a server on Amazon Web Services, here are instructions for setting up an `EC2 gpu server with Theano`_.

  * CuDNN_ (optional but recommended for CNN's): for a fast convolution support from Nvidia. You will want to move the files to Theano's directory like the instructions say here: `Theano cuDNN integration`_.

* `Six`_: Python 2/3 compatibility library.

* `Pillow (PIL)`_ (optional): image manipulation functionality.

* PyYAML_ (optional): used for YAML parsing of config files.

* Bokeh_ (optional): if you want live charting/plotting of values during training or testing.

* NLTK_ (optional): if you want nlp functions like word tokenization.

All of these Python dependencies (not the system-specific ones like CUDA or HDF5), can be installed with :code:`pip install -r requirements.txt` inside the root OpenDeep folder.

.. _installation instructions here: http://deeplearning.net/software/theano/install.html#bleeding-edge-install-instructions
.. _CUDA drivers: https://developer.nvidia.com/cuda-toolkit
.. _instructions for setting up Theano for the GPU: http://deeplearning.net/software/theano/tutorial/using_gpu.html
.. _EC2 gpu server with Theano: http://markus.com/install-theano-on-aws
.. _CuDNN: https://developer.nvidia.com/cuDNN
.. _Theano cuDNN integration: http://deeplearning.net/software/theano/library/sandbox/cuda/dnn.html
.. _Pillow (PIL): https://pillow.readthedocs.org/installation.html
.. _Six: https://pythonhosted.org/six/
.. _PyYAML: http://pyyaml.org/
.. _Bokeh: http://bokeh.pydata.org/en/latest/
.. _NLTK: http://www.nltk.org/

Install from source
-------------------
1) Navigate to your desired installation directory and download the github repository::

    git clone https://github.com/vitruvianscience/opendeep.git

2) Navigate to the top-level folder (should be named OpenDeep and contain the file setup.py) and run setup.py with develop mode::

    cd opendeep
    python setup.py develop

Using :code:`python setup.py develop` instead of the normal :code:`python setup.py install` allows you to update the repository files by pulling
from git and have the whole package update! No need to reinstall when you get the latest files.

That's it! Now you should be able to import opendeep into python modules.

More Information
================
Source code: https://github.com/vitruvianscience/opendeep

Documentation and tutorials: http://www.opendeep.org/

User group: `opendeep-users`_

Developer group: `opendeep-dev`_

Twitter: `@opendeep`_

We would love all help to make this the best library possible! Feel free to fork the repository and
join the Google groups!

.. _opendeep-users: https://groups.google.com/forum/#!forum/opendeep-users/
.. _opendeep-dev: https://groups.google.com/forum/#!forum/opendeep-dev/
.. _@opendeep: https://twitter.com/opendeep


Why OpenDeep?
=============

- **Modularity**. A lot of recent deep learning progress has come from combining multiple models. Existing libraries are either too confusing or not easily extensible enough to perform novel research and also quickly set up existing algorithms at scale. This need for transparency and modularity is the main motivating factor for creating the OpenDeep library, where we hope novel research and industry use can both be easily implemented.

- **Ease of use**. Many libraries require a lot of familiarity with deep learning or their specific package structures. OpenDeep's goal is to be the best-documented deep learning library and have smart enough default code that someone without a background can start training models, while experienced practitioners can easily create and customize their own algorithms.

- **State of the art**. A side effect of modularity and ease of use, OpenDeep aims to maintain state-of-the-art performance as new algorithms and papers get published. As a research library, citing and accrediting those authors and code used is very important to the library.
