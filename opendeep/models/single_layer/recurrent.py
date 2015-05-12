"""
This module provides a framework for constructing recurrent networks.

.. todo::
    Implement!

"""
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
# internal references
from opendeep.models.model import Model

log = logging.getLogger(__name__)


class Recurrent(Model):
    """
    Your run-of-the-mill recurrent model. Normally not as good as LSTM/GRU, but it is simplest.
    """
    def __init__(self):
        # TODO: recurrent model
        log.error("Recurrent not implemented!")
        raise NotImplementedError("Recurrent not implemented!")
