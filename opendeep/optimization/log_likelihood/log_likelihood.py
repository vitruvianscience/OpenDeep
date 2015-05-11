'''
General interface for creating a log-likelihood score object.
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging

log = logging.getLogger(__name__)

class LogLikelihood(object):
    '''
    Default interface for a log-likelihood score
    '''
    def __init__(self):
        #TODO: Make LogLikehood interface
        raise NotImplementedError("No LogLikelihood interface created yet!")