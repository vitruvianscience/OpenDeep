'''
Object for the Piano-midi.de midi dataset
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import logging
# internal imports
from opendeep.data.dataset import FileDataset

log = logging.getLogger(__name__)

class PianoMidiDe(FileDataset):
    '''
    Object for the Piano-midi.de midi dataset. Pickled file of midi piano roll provided by Montreal's Nicolas Boulanger-Lewandowski into train, valid, and test sets.
    '''
    def __init__(self, dataset_dir='../../datasets'):
        filename = 'Piano-midi.de.zip'
        source = 'http://www-etud.iro.umontreal.ca/~boulanni/Piano-midi.de.zip'
        super(PianoMidiDe, self).__init__(filename=filename, source=source, dataset_dir=dataset_dir)