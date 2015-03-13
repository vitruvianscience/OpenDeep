'''
Unit testing for stochastic gradient descent
See the framework discussed in:
'Unit Tests for Stochastic Optimization'
Tom Schaul, Ioannis Antonoglou, David Silver
http://arxiv.org/abs/1312.6055
https://github.com/IoannisAntonoglou/optimBench
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import unittest
import logging
# third party libraries
import theano.tensor as T
# internal references
import opendeep.log.logger as logger
from opendeep.optimization.stochastic_gradient_descent import SGD

class TestSGD(unittest.TestCase):

    def setUp(self):
        # configure the root logger
        logger.config_root_logger()
        # get a logger for this session
        self.log = logging.getLogger(__name__)

    def testSimple(self):
        self.A = T.fmatrix('A')
        self.X = T.fmatrix('X')
        self.B = T.fmatrix('B')

    def tearDown(self):
        pass