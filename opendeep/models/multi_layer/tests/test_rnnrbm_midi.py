# produce results from here:
# http://deeplearning.net/tutorial/rnnrbm.html

from __future__ import print_function
import numpy
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from opendeep.models.multi_layer.rnn_rbm import RNN_RBM
from opendeep.data.standard_datasets.midi.nottingham import Nottingham
from opendeep.data.standard_datasets.midi.jsb_chorales import JSBChorales
from opendeep.data.standard_datasets.midi.musedata import MuseData
from opendeep.data.standard_datasets.midi.piano_midi_de import PianoMidiDe
from opendeep.optimization.stochastic_gradient_descent import SGD
from opendeep.optimization.adadelta import AdaDelta
from opendeep.monitor.plot import Plot
from opendeep.utils.image import tile_raster_images
from opendeep.utils.misc import closest_to_square_factors
from opendeep.utils.midi import midiwrite
from opendeep.monitor.monitor import Monitor
import PIL.Image as Image
try:
    import pylab
    has_pylab = True
except ImportError:
    print("pylab isn't available.")
    print("It can be installed with 'pip install -q Pillow'")
    has_pylab = False

import logging
from opendeep.log.logger import config_root_logger

log = logging.getLogger(__name__)


def run_midi(dataset):
    log.info("Creating RNN-RBM for dataset %s!", dataset)

    outdir = "outputs/rnnrbm/%s/" % dataset

    # grab the MIDI dataset
    if dataset == 'nottingham':
        midi = Nottingham()
    elif dataset == 'jsb':
        midi = JSBChorales()
    elif dataset == 'muse':
        midi = MuseData()
    elif dataset == 'piano_de':
        midi = PianoMidiDe()
    else:
        raise AssertionError("dataset %s not recognized." % dataset)

    # create the RNN-RBM
    # rng = numpy.random
    # rng.seed(0xbeef)
    # mrg = RandomStreams(seed=rng.randint(1 << 30))
    rng = numpy.random.RandomState(1234)
    mrg = RandomStreams(rng.randint(2 ** 30))
    # rnnrbm = RNN_RBM(input_size=88,
    #                  hidden_size=150,
    #                  rnn_hidden_size=100,
    #                  k=15,
    #                  weights_init='gaussian',
    #                  weights_std=0.01,
    #                  rnn_weights_init='gaussian',
    #                  rnn_weights_std=0.0001,
    #                  rng=rng,
    #                  outdir=outdir)
    rnnrbm = RNN_RBM(input_size=88,
                     hidden_size=150,
                     rnn_hidden_size=100,
                     k=15,
                     weights_init='gaussian',
                     weights_std=0.01,
                     rnn_weights_init='identity',
                     rnn_hidden_activation='relu',
                     # rnn_weights_init='gaussian',
                     # rnn_hidden_activation='tanh',
                     rnn_weights_std=0.0001,
                     mrg=mrg,
                     outdir=outdir)

    # make an optimizer to train it
    optimizer = SGD(model=rnnrbm,
                    dataset=midi,
                    n_epoch=200,
                    batch_size=100,
                    minimum_batch_size=2,
                    learning_rate=.001,
                    save_frequency=10,
                    early_stop_length=200,
                    momentum=False,
                    momentum_decay=False,
                    nesterov_momentum=False)

    optimizer = AdaDelta(model=rnnrbm,
                         dataset=midi,
                         n_epoch=200,
                         batch_size=100,
                         minimum_batch_size=2,
                         # learning_rate=1e-4,
                         learning_rate=1e-6,
                         save_frequency=10,
                         early_stop_length=200)

    ll = Monitor('pseudo-log', rnnrbm.get_monitors()['pseudo-log'], test=True)
    mse = Monitor('frame-error', rnnrbm.get_monitors()['mse'], valid=True, test=True)

    plot = Plot(bokeh_doc_name='rnnrbm_midi_%s' % dataset, monitor_channels=[ll, mse], open_browser=True)

    # perform training!
    optimizer.train(plot=plot)
    # use the generate function!
    generated, _ = rnnrbm.generate(initial=None, n_steps=200)

    dt = 0.3
    r = (21, 109)
    midiwrite(outdir + 'rnnrbm_generated_midi.mid', generated, r=r, dt=dt)

    if has_pylab:
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
    image.save(outdir + 'rnnrbm_midi_weights.png')

    log.debug("done!")
    del midi
    del rnnrbm
    del optimizer

    # if has_pylab:
    #     pylab.show()

if __name__ == '__main__':
    config_root_logger()
    run_midi('jsb')
    run_midi('piano_de')
    run_midi('muse')
    run_midi('nottingham')