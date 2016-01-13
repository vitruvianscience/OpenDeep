# standard libraries
import unittest
import logging
import os
import shutil
# third party
import numpy
# internal references
from opendeep.data.dataset_file import FileDataset
from opendeep.utils.file_ops import mkdir_p


class TestFileDataset(unittest.TestCase):

    def setUp(self):
        # get a logger for this session
        self.log = logging.getLogger(__name__)
        self.dir = "filedataset_test_files"
        self.single_file_dir = os.path.join(self.dir, "target_in_file")
        mkdir_p(self.single_file_dir)
        # create files
        with open(os.path.join(self.single_file_dir, "1.txt"), 'w') as f:
            f.write("1,2,3,4,5\t0\n6,7,8,9,0\t1")
        with open(os.path.join(self.single_file_dir, "1.txt"), 'r') as f:
            for line in f:
                print list(line)

    def testSingleFile(self):
        pass

    def tearDown(self):
        shutil.rmtree(self.dir)

if __name__ == '__main__':
    unittest.main()
