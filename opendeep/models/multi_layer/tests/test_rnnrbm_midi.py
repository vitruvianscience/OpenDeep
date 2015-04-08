"""
produce results from here:
http://deeplearning.net/tutorial/rnnrbm.html
"""

import numpy
from theano.tensor.shared_randomstreams import RandomStreams
from opendeep.models.multi_layer.rnn_rbm import RNN_RBM
from opendeep.data.standard_datasets.midi.nottingham import Nottingham
from opendeep.optimization.stochastic_gradient_descent import SGD
from opendeep.utils.image import tile_raster_images
from opendeep.utils.misc import closest_to_square_factors
from opendeep.utils.midi import midiwrite
import PIL.Image as Image
import pylab


if __name__ == '__main__':
    # set up the logging environment to display outputs (optional)
    # although this is recommended over print statements everywhere
    import logging
    import opendeep.log.logger as logger
    logger.config_root_logger()
    log = logging.getLogger(__name__)
    log.info("Creating RNN-RBM!")

    # grab the MNIST dataset
    midi = Nottingham()
    # create the RNN-RBM
    numpy.random.seed(0xbeef)
    rng = RandomStreams(seed=numpy.random.randint(1 << 30))
    rnnrbm = RNN_RBM(input_size=88,
                     hidden_size=150,
                     recurrent_hidden_size=100,
                     k=15,
                     weights_init='gaussian',
                     weights_std=0.01,
                     recurrent_weights_init='gaussian',
                     recurrent_weights_std=0.0001,
                     rng=rng)
    # make an optimizer to train it
    optimizer = SGD(model=rnnrbm,
                    dataset=midi,
                    n_epoch=200,
                    batch_size=100,
                    minimum_batch_size=2,
                    learning_rate=.001,
                    save_frequency=10,
                    momentum=False,
                    nesterov_momentum=False)
    # perform training!
    optimizer.train()
    # use the generate function!
    generated, _ = rnnrbm.generate(initial=None, n_steps=200)

    dt = 0.3
    r = (21, 109)
    midiwrite('rnnrbm_generated_midi.mid', generated, r=r, dt=dt)
    extent = (0, dt * len(generated)) + r
    pylab.figure()
    pylab.imshow(generated.T, origin='lower', aspect='auto',
                 interpolation='nearest', cmap=pylab.cm.gray_r,
                 extent=extent)
    pylab.xlabel('time (s)')
    pylab.ylabel('MIDI note number')
    pylab.title('generated piano-roll')

    # Construct image from the weight matrix
    image = Image.fromarray(
        tile_raster_images(
            X=rnnrbm.W.get_value(borrow=True).T,
            img_shape=closest_to_square_factors(rnnrbm.input_size),
            tile_shape=closest_to_square_factors(rnnrbm.hidden_size),
            tile_spacing=(1, 1)
        )
    )
    image.save('rnnrbm_midi_weights.png')

    print "done!"
    del midi
    del rnnrbm
    del optimizer

    pylab.show()