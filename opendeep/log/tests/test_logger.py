'''
Unit testing for the logger class
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import unittest
import os
import sys
from StringIO import StringIO
import logging
# internal references
import opendeep.log.logger as logger

class TestLogger(unittest.TestCase):

    def setUp(self):
        # change sys.stdout to StringIO for the duration of the test to test console output
        self.saved_stdout = sys.stdout
        self.out = StringIO()
        sys.stdout = self.out
        # configure the root logger
        logger.config_root_logger()
        # get a logger for this session
        self.log = logging.getLogger(__name__)
        # set the paths for the logs (comes from reading logging_config.json)
        self.error_path = '../logs/error/errors.log'
        self.info_path  = '../logs/info/info.log'
        self.paths = [self.error_path, self.info_path]

    def testLogFilesExist(self):
        for path in self.paths:
            assert os.path.exists(path)

    def testDebugConsole(self):
        self.log.debug('Unit testing debug.')
        output = self.out.getvalue().strip()
        assert 'Unit testing debug.' in output

    def testInfoFile(self):
        self.log.info('Unit testing info.')
        self.log.warning('Unit testing warning.')
        with open(self.info_path,'r') as f:
            text = f.readlines()
        # should be last lines in file
        assert 'Unit testing info.' in text[-2]
        assert 'Unit testing warning.' in text[-1]

    def testErrorFile(self):
        self.log.error('Unit testing error.')
        self.log.critical('Unit testing critical.')
        with open(self.error_path,'r') as f:
            text = f.readlines()
        # should be last lines in file
        assert 'Unit testing error.' in text[-2]
        assert 'Unit testing critical.' in text[-1]

    def tearDown(self):
        sys.stdout = self.saved_stdout


if __name__ == '__main__':
    unittest.main()