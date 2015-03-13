"""
.. module:: recursive

This module provides the general framework for recursive networks
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

class Recursive(Model):
    # TODO: recursive model for things like language or scene parsing
    log.error("Recursive not implemented!")
    raise NotImplementedError("Recursive not implemented!")