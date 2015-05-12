# standard libraries
from __future__ import print_function
import unittest
import numpy
# internal references
from opendeep.data.standard_datasets.midi.musedata import MuseData
from opendeep.data.standard_datasets.midi.jsb_chorales import JSBChorales
from opendeep.data.standard_datasets.midi.nottingham import Nottingham
from opendeep.data.standard_datasets.midi.piano_midi_de import PianoMidiDe
from opendeep.data.dataset import TRAIN, VALID, TEST


class TestMuse(unittest.TestCase):

    def setUp(self):
        print("setting up!")
        # get the muse dataset
        self.muse = MuseData()
        # get the jsb dataset
        # self.jsb = JSBChorales()
        # get nottingham dataset
        self.nottingham = Nottingham()
        # get the piano-midi-de dataset
        # self.piano = PianoMidiDe()


    def testSizes(self):
        print('muse train %s' % str(self.muse.train.shape.eval()[0]))
        assert self.muse.train.shape.eval()[0] == numpy.sum([l[0] for l in self.muse.getDataShape(TRAIN)])
        assert self.muse.valid.shape.eval()[0] == numpy.sum([l[0] for l in self.muse.getDataShape(VALID)])
        assert self.muse.test.shape.eval()[0]  == numpy.sum([l[0] for l in self.muse.getDataShape(TEST)])

        print('nottingham train %s' % str(self.nottingham.train.shape.eval()[0]))
        assert self.nottingham.train.shape.eval()[0] == numpy.sum([l[0] for l in self.nottingham.getDataShape(TRAIN)])
        assert self.nottingham.valid.shape.eval()[0] == numpy.sum([l[0] for l in self.nottingham.getDataShape(VALID)])
        assert self.nottingham.test.shape.eval()[0]  == numpy.sum([l[0] for l in self.nottingham.getDataShape(TEST)])

    def tearDown(self):
        del self.muse
        # del self.jsb
        del self.nottingham
        # del self.piano
        print("done!")


if __name__ == '__main__':
    unittest.main()