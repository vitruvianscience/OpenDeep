"""
General interface for creating a log-likelihood score object.
"""
# standard libraries
import logging

log = logging.getLogger(__name__)

class LogLikelihood(object):
    """
    Default interface for a log-likelihood score
    """
    def __init__(self):
        #TODO: Make LogLikehood interface
        raise NotImplementedError("No LogLikelihood interface created yet!")
