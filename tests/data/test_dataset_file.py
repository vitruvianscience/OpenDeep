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
        self.data0 = [[1,2,3,4,5],[2,3,4,5,6]]
        self.data1 = [[6,7,8,9,0],[7,8,9,0,1]]
        with open(os.path.join(self.single_file_dir, "1.txt"), 'w') as f:
            f.write("1,2,3,4,5\t0\n6,7,8,9,0\t1")
        with open(os.path.join(self.single_file_dir, "2.txt"), 'w') as f:
            f.write("2,3,4,5,6\t0\n7,8,9,0,2\t1")

        self.data_files = os.path.join(self.dir, "target_in_filename")
        mkdir_p(self.data_files)
        # create files
        self.cat = "I am a feline! \nMeow."
        self.dog = "I am a canine! \nWoof."
        with open(os.path.join(self.data_files, "cat.txt"), 'w') as f:
            f.write(self.cat)
        with open(os.path.join(self.data_files, "dog.txt"), 'w') as f:
            f.write(self.dog)

    def testSingleFile(self):
        data = FileDataset(path=self.single_file_dir, inputs_preprocess=lambda x: numpy.array(x.split("\t")[0]), )

    def testFilename(self):
        pass

    def tearDown(self):
        shutil.rmtree(self.dir)

if __name__ == '__main__':
    unittest.main()
