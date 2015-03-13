"""
.. module:: recurrent

This module provides different recurrent networks
"""
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import logging
# internal references
from opendeep.models.model import Model

log = logging.getLogger(__name__)

class Recurrent(Model):
    """
    Your run-of-the-mill recurrent model. Normally not as good as LSTM/GRU, but it is simplest.
    """
    # TODO: recurrent model
    log.error("Recurrent not implemented!")
    raise NotImplementedError("Recurrent not implemented!")

class LSTM(Model):
    """
    Long short-term memory units.
    https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/lstm.py
    """
    # TODO: LSTM recurrent model
    log.error("LSTM not implemented!")
    raise NotImplementedError("LSTM not implemented!")

class GRU(Model):
    """
    Gated recurrent units.
    """
    # TODO: GRU recurrent model
    log.error("GRU not implemented!")
    raise NotImplementedError("GRU not implemented!")