This is the list of items currently wanted in the package - if you finish one, please remove from the list!

Datasets
========
* Scaling/normalization support for cleaning datasets.
* Database connection support.
* Large dataset formats of numpy.memmap, h5py, pytables.
* Spark support.

Models
======
(also see opendeep.models.future package)

* Modular RNN framework support, ideas from "How to Construct Deep Recurrent Neural Networks" http://arxiv.org/abs/1312.6026  This framework is key for working with many other model types.
* GRU and LSTM for RNN hidden unit as a mixin.
* Memory networks http://arxiv.org/abs/1410.3916 and http://arxiv.org/abs/1503.08895
* 1D convolutional networks (temporal cnn)
* CNN-RNN (image-captioning)
* Stacking autoencoders
* DBN
* Support for multiple outputs from a model
* Support for multiple inputs to a model
* Maxout network
* Deep Q Network
* RL-NTM http://arxiv.org/abs/1505.00521
* Highway networks http://arxiv.org/abs/1505.00387

Optimization
============
* ADAM https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-chilimbi.pdf
* Target Propagation http://arxiv.org/abs/1407.7906 and http://arxiv.org/abs/1412.7525
* Distributed learning methods (look in opendeep.distributed.references.txt for reference papers.)
* Unsupervised sparsity optimization http://arxiv.org/abs/1402.5766

Misc
====
* Visualization module! Support t-sne, activation maps, etc.
* Regularization mixin for training costs: l1, l2
* Log-likelihood estimators.
* Bayesian hyperparameter optimization
* Monitor outservice to a database
* Add the ability to create a Prototype from a config file.