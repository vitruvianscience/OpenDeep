# standard libraries
from __future__ import print_function
import unittest
# internal references
from opendeep.data.standard_datasets.midi.musedata import MuseData
from opendeep.data.standard_datasets.midi.jsb_chorales import JSBChorales
from opendeep.data.standard_datasets.midi.nottingham import Nottingham
from opendeep.data.standard_datasets.midi.piano_midi_de import PianoMidiDe
from opendeep.log.logger import config_root_logger


class TestMidi(unittest.TestCase):

    def setUp(self):
        print("setting up!")
        config_root_logger()
        # get the muse dataset
        self.muse = MuseData(path='../../../datasets/MuseData')
        # get the jsb dataset
        self.jsb = JSBChorales(path='../../../datasets/JSB Chorales')
        # get nottingham dataset
        self.nottingham = Nottingham(path='../../../datasets/Nottingham')
        # get the piano-midi-de dataset
        self.piano = PianoMidiDe(path='../../../datasets/Piano-midi.de')

    def testSizes(self):
        pass

    def tearDown(self):
        del self.muse
        del self.jsb
        del self.nottingham
        del self.piano
        print("done!")


if __name__ == '__main__':
    unittest.main()
