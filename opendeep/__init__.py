"""
Code used throughout the entire OpenDeep package.
"""
from __future__ import division, absolute_import, print_function

# get the package version number from a single location
try:
    from . import version
    v = version.__version__
except ImportError:
    import os
    here = os.path.abspath(os.path.dirname(__file__))
    versionfile = os.path.join(here, 'version.py')
    # Grab the appropriate version number from opendeep/version.py so we only have to keep track of it in one place!
    exec(compile(open(versionfile).read(), versionfile, 'exec'))
    # now we have the appropriate version in __version__
    v = __version__  # it is there, trust me :) IDE's won't recognize that exec does anything.

__version__ = v
__copyright__ = "Copyright 2015, Vitruvian Science"
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# internal imports for easy package structure
from . import data
from . import log
from . import data
from . import models
from . import monitor
from . import optimization
from . import utils

# so we can get `from opendeep import function, grad, sharedX,` etc.
from .utils.constructors import *

# so we can config logger from the base import
from .log import config_root_logger