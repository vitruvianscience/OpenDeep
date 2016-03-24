from __future__ import division
import unittest
from types import FunctionType
import numpy as np
from opendeep.utils.initializers import (UniformIntervalFunc, Uniform, Gaussian, Identity, Orthogonal, Constant)


class TestUniform(unittest.TestCase):
    def setUp(self):
        pass

    def testString(self):
        i = Uniform('glorot')
        f = UniformIntervalFunc.glorot
        self.assertIsInstance(i.interval_func, FunctionType)
        self.assertEquals(
            i.interval_func, f, "The function grabbed by Uniform('glorot') does not match UniformIntervalFunc.glorot"
        )
        self.assertRaises(AttributeError, Uniform, interval='most_definitely_not_a_func_foobar_testing')

    def testSingleNumber(self):
        def checkN(n):
            i = Uniform(n)
            self.assertEquals(i.min, -1*abs(n))
            self.assertEquals(i.max, abs(n))
        checkN(int(42))
        checkN(int(-42))
        checkN(long(42))
        checkN(float(42))
        checkN(np.int8(42))
        checkN(np.int64(42))
        checkN(np.float16(42))

    def testTuple(self):
        def checkTup(tup):
            if len(tup) == 1:
                i = Uniform(tup)
                self.assertEquals(i.min, -1*abs(tup[0]))
                self.assertEquals(i.max, abs(tup[0]))
            elif len(tup) < 1:
                self.assertRaises(AttributeError, Uniform, interval=tup)
            else:
                all_nums = True
                for ele in tup:
                    if not isinstance(ele, (int, float, long, np.number)):
                        all_nums = False
                if all_nums:
                    i = Uniform(tup)
                    self.assertEquals(i.min, np.min(tup))
                    self.assertEquals(i.max, np.max(tup))
                else:
                    self.assertRaises(AttributeError, Uniform, interval=tup)
        checkTup([])
        checkTup([-42])
        checkTup([42])
        checkTup([1, 5])
        checkTup([-1, 5])
        checkTup([5, 5])
        checkTup([5, 1])
        checkTup([5, 'a'])
        checkTup(['b', 'a'])
        checkTup([float(3), 1])
        checkTup([float(-3), np.int8(1)])
        checkTup([5, 1, 3])

    def testFunction(self):
        def checkFunc(f):
            i = Uniform(f)

        checkFunc(lambda shape: shape[0])

    def testBadType(self):
        self.assertRaises(AttributeError, Uniform, interval=None)
        self.assertRaises(AttributeError, Uniform, interval={"min": 1, "max": 2})
        self.assertRaises(AttributeError, Uniform, interval=[])
    #TODO: somehow test the uniform distribution?

    def tearDown(self):
        pass


class TestGaussian(unittest.TestCase):
    def setUp(self):
        pass

    def testInit(self):
        i = Gaussian(mean=0, std=1)
        self.assertEquals(i.mean, 0)
        self.assertEquals(i.std, 1)
    #TODO: somehow test the gaussian distribution?

    def tearDown(self):
        pass


class TestIdentity(unittest.TestCase):
    def setUp(self):
        pass

    def testInit(self):
        i = Identity(add_noise=None)
    #TODO: somehow test the identity matrix?

    def tearDown(self):
        pass


class TestOrthogonal(unittest.TestCase):
    def setUp(self):
        pass

    def testInit(self):
        i = Orthogonal()
    #TODO: somehow test the orthogonal distribution?

    def tearDown(self):
        pass


class TestConstant(unittest.TestCase):
    def setUp(self):
        pass

    def testSingleVal(self):
        def checkConstant(n):
            i = Constant(n)
            b = i((5,))
            for e in b.eval():
                self.assertEquals(e, n)
        checkConstant(0)
        checkConstant(-1)
        checkConstant(5)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
