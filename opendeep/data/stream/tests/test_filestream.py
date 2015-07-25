import unittest
import shutil
from opendeep.utils.file_ops import mkdir_p
from opendeep.data.stream.filestream import FileStream


class TestFilestream(unittest.TestCase):

    def setUp(self):
        # create test directories and files
        self.base = "test_filestream_dir/"
        train = self.base+"train/"
        valid = self.base+"valid/"
        test = self.base+"test/"
        mkdir_p(train)
        mkdir_p(valid)
        mkdir_p(test)
        # some files
        with open(train+"train1.txt", "w") as f:
            f.write("TRAIN1a\ntrain1b\n\n")
        with open(train + "train2.txt", "w") as f:
            f.write("TRAIN2a\ntrain2b\n\n")
        with open(train + "train3.md", "w") as f:
            f.write("TRAIN3a\ntrain3b\n\n")

        with open(valid + "valid1.txt", "w") as f:
            f.write("valid1a\nvalid1b\n\n")
        with open(valid + "valid2.txt", "w") as f:
            f.write("valid2a\nvalid2b\n\n")
        with open(valid + "valid3.md", "w") as f:
            f.write("valid3a\nvalid3b\n\n")

        with open(test + "test1.txt", "w") as f:
            f.write("test1a\ntest1b\n\n")
        with open(test + "test2.txt", "w") as f:
            f.write("test2a\ntest2b\n\n")
        with open(test + "test3.md", "w") as f:
            f.write("test3a\ntest3b\n\n")

    def testRecursive(self):
        fs = FileStream(path=self.base)
        check_lines = ['test3a\n',
                       'test3b\n',
                       '\n',
                       'test2a\n',
                       'test2b\n',
                       '\n',
                       'test1a\n',
                       'test1b\n',
                       '\n',
                       'TRAIN3a\n',
                       'train3b\n',
                       '\n',
                       'TRAIN2a\n',
                       'train2b\n',
                       '\n',
                       'TRAIN1a\n',
                       'train1b\n',
                       '\n',
                       'valid2a\n',
                       'valid2b\n',
                       '\n',
                       'valid1a\n',
                       'valid1b\n',
                       '\n',
                       'valid3a\n',
                       'valid3b\n',
                       '\n']
        for line in fs:
            assert line in check_lines, "Found extra line: %s" % [str(line)]
            check_lines.remove(line)
        assert len(check_lines) == 0, "Didn't catch all lines, still have: %s" % str(check_lines)

        # check that it is reusable
        check_lines = ['test3a\n',
                       'test3b\n',
                       '\n',
                       'test2a\n',
                       'test2b\n',
                       '\n',
                       'test1a\n',
                       'test1b\n',
                       '\n',
                       'TRAIN3a\n',
                       'train3b\n',
                       '\n',
                       'TRAIN2a\n',
                       'train2b\n',
                       '\n',
                       'TRAIN1a\n',
                       'train1b\n',
                       '\n',
                       'valid2a\n',
                       'valid2b\n',
                       '\n',
                       'valid1a\n',
                       'valid1b\n',
                       '\n',
                       'valid3a\n',
                       'valid3b\n',
                       '\n']
        for line in fs:
            assert line in check_lines, "Found extra line: %s" % [str(line)]
            check_lines.remove(line)
        assert len(check_lines) == 0, "Didn't catch all lines, still have: %s" % str(check_lines)

    def testFilters(self):
        fs = FileStream(path=self.base, filter=".*train/.*")
        check_lines = ['TRAIN3a\n',
                       'train3b\n',
                       '\n',
                       'TRAIN2a\n',
                       'train2b\n',
                       '\n',
                       'TRAIN1a\n',
                       'train1b\n',
                       '\n']
        for line in fs:
            assert line in check_lines, "Found extra line: %s" % [str(line)]
            check_lines.remove(line)
        assert len(check_lines) == 0, "Didn't catch all lines, still have: %s" % str(check_lines)

        fs = FileStream(path=self.base, filter=".*train/.*.txt")
        check_lines = ['TRAIN2a\n',
                       'train2b\n',
                       '\n',
                       'TRAIN1a\n',
                       'train1b\n',
                       '\n']
        for line in fs:
            assert line in check_lines, "Found extra line: %s" % [str(line)]
            check_lines.remove(line)
        assert len(check_lines) == 0, "Didn't catch all lines, still have: %s" % str(check_lines)

        fs = FileStream(path=self.base, filter=".*trololo")
        check_lines = []
        for line in fs:
            assert line in check_lines, "Found extra line: %s" % [str(line)]
            check_lines.remove(line)
        assert len(check_lines) == 0, "Didn't catch all lines, still have: %s" % str(check_lines)

    def testPreprocess(self):
        # lowercase
        fs = FileStream(path=self.base, preprocess=lambda s: s.lower())
        check_lines = ['test3a\n',
                       'test3b\n',
                       '\n',
                       'test2a\n',
                       'test2b\n',
                       '\n',
                       'test1a\n',
                       'test1b\n',
                       '\n',
                       'train3a\n',
                       'train3b\n',
                       '\n',
                       'train2a\n',
                       'train2b\n',
                       '\n',
                       'train1a\n',
                       'train1b\n',
                       '\n',
                       'valid2a\n',
                       'valid2b\n',
                       '\n',
                       'valid1a\n',
                       'valid1b\n',
                       '\n',
                       'valid3a\n',
                       'valid3b\n',
                       '\n']
        for line in fs:
            assert line in check_lines, "Found extra line: %s" % [str(line)]
            check_lines.remove(line)
        assert len(check_lines) == 0, "Didn't catch all lines, still have: %s" % str(check_lines)

        # tokenize by character
        fs = FileStream(path=self.base, preprocess=lambda s: list(s))
        check_lines = ['test3a\n',
                       'test3b\n',
                       '\n',
                       'test2a\n',
                       'test2b\n',
                       '\n',
                       'test1a\n',
                       'test1b\n',
                       '\n',
                       'TRAIN3a\n',
                       'train3b\n',
                       '\n',
                       'TRAIN2a\n',
                       'train2b\n',
                       '\n',
                       'TRAIN1a\n',
                       'train1b\n',
                       '\n',
                       'valid2a\n',
                       'valid2b\n',
                       '\n',
                       'valid1a\n',
                       'valid1b\n',
                       '\n',
                       'valid3a\n',
                       'valid3b\n',
                       '\n']
        lines = []
        [lines.extend(list(line)) for line in check_lines]
        check_lines = lines
        for line in fs:
            assert line in check_lines, "Found extra line: %s" % [str(line)]
            check_lines.remove(line)
        assert len(check_lines) == 0, "Didn't catch all lines, still have: %s" % str(check_lines)

    def testFuture(self):
        fs = FileStream(path=self.base, n_future=1)
        check_lines = ['test3b\n',
                       '\n',
                       'test2a\n',
                       'test2b\n',
                       '\n',
                       'test1a\n',
                       'test1b\n',
                       '\n',
                       'TRAIN3a\n',
                       'train3b\n',
                       '\n',
                       'TRAIN2a\n',
                       'train2b\n',
                       '\n',
                       'TRAIN1a\n',
                       'train1b\n',
                       '\n',
                       'valid2a\n',
                       'valid2b\n',
                       '\n',
                       'valid1a\n',
                       'valid1b\n',
                       '\n',
                       'valid3a\n',
                       'valid3b\n',
                       '\n']
        for line in fs:
            assert line in check_lines, "Found extra line: %s" % [str(line)]
            check_lines.remove(line)
        assert len(check_lines) == 0, "Didn't catch all lines, still have: %s" % str(check_lines)

        fs = FileStream(path=self.base, n_future=4)
        check_lines = ['test2b\n',
                       '\n',
                       'test1a\n',
                       'test1b\n',
                       '\n',
                       'TRAIN3a\n',
                       'train3b\n',
                       '\n',
                       'TRAIN2a\n',
                       'train2b\n',
                       '\n',
                       'TRAIN1a\n',
                       'train1b\n',
                       '\n',
                       'valid2a\n',
                       'valid2b\n',
                       '\n',
                       'valid1a\n',
                       'valid1b\n',
                       '\n',
                       'valid3a\n',
                       'valid3b\n',
                       '\n']
        for line in fs:
            assert line in check_lines, "Found extra line: %s" % [str(line)]
            check_lines.remove(line)
        assert len(check_lines) == 0, "Didn't catch all lines, still have: %s" % str(check_lines)

    def tearDown(self):
            shutil.rmtree(self.base)


if __name__ == '__main__':
    unittest.main()
