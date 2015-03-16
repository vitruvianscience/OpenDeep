'''
Unit testing for the midi datasets
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import unittest
import logging
# internal references
from opendeep.data.standard_datasets.midi.musedata import MuseData
from opendeep.data.standard_datasets.midi.jsb_chorales import JSBChorales
from opendeep.data.standard_datasets.midi.nottingham import Nottingham
from opendeep.data.standard_datasets.midi.piano_midi_de import PianoMidiDe
import opendeep.data.dataset as dataset
import opendeep.log.logger as logger
from opendeep.data.iterators.sequential import SequentialIterator
from opendeep.data.iterators.random import RandomIterator


class TestMuse(unittest.TestCase):

    def setUp(self):
        # configure the root logger
        logger.config_root_logger()
        # get a logger for this session
        self.log = logging.getLogger(__name__)
        # get the muse dataset
        self.muse = MuseData()
        # get the jsb dataset
        self.jsb = JSBChorales()
        # get nottingham dataset
        self.nottingham = Nottingham()
        # get the piano-midi-de dataset
        self.piano = PianoMidiDe()


    def testSizes(self):
        pass


    def tearDown(self):
        del self.muse
        del self.jsb
        del self.nottingham
        del self.piano


if __name__ == '__main__':
    unittest.main()