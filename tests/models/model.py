# standard libraries
import unittest
# third party
import numpy as np
from theano.tensor import matrix, dot
# internal references
from opendeep.log.logger import config_root_logger
from opendeep.models import Model, Prototype, Dense
from opendeep.utils.weights import get_weights
from opendeep.optimization import SGD
from opendeep.optimization.loss import BinaryCrossentropy
from opendeep.data import NumpyDataset


class TestParamSharing(unittest.TestCase):

    def setUp(self):
        # configure the root logger
        # config_root_logger()
        pass

    def testAutoEncoder(self):
        try:
            s = (None, 3)
            x = matrix('xs')
            e = Dense(inputs=(s, x), outputs=int(s[1]*2), activation='sigmoid')
            W = e.get_param("W")
            d = Dense(inputs=e, outputs=s[1], params={'W': W.T}, activation='sigmoid')
            ae = Prototype([e, d])

            x2 = matrix('xs1')
            W2 = d.get_param("W")
            e2 = Dense(inputs=(s, x2), outputs=int(s[1]*2), params={"W": W2.T, "b": e.get_param('b')}, activation='sigmoid')
            W3 = e2.get_param("W")
            d2 = Dense(inputs=e2, outputs=s[1], params={"W": W3.T, 'b': d.get_param('b')}, activation='sigmoid')
            ae2 = Prototype([e2, d2])

            aerun1 = ae.run(np.array([[.1,.5,.9]], dtype='float32'))
            ae2run1 = ae.run(np.array([[.1,.5,.9]], dtype='float32'))
            self.assertTrue(np.array_equal(aerun1, ae2run1))

            data = np.ones((10,3), dtype='float32')*.1
            data = np.vstack([data, np.ones((10,3), dtype='float32')*.2])
            data = np.vstack([data, np.ones((10,3), dtype='float32')*.3])
            data = np.vstack([data, np.ones((10,3), dtype='float32')*.4])
            data = np.vstack([data, np.ones((10,3), dtype='float32')*.5])
            data = np.vstack([data, np.ones((10,3), dtype='float32')*.6])
            data = np.vstack([data, np.ones((10,3), dtype='float32')*.7])
            data = np.vstack([data, np.ones((10,3), dtype='float32')*.8])
            data = np.vstack([data, np.ones((10,3), dtype='float32')*.9])
            data = np.vstack([data, np.ones((10,3), dtype='float32')*0])
            dataset = NumpyDataset(data)
            sgd = SGD(dataset=dataset, model=ae, loss=BinaryCrossentropy(inputs=ae.get_outputs(), targets=x), epochs=5)
            sgd.train()

            aerun2 = ae.run(np.array([[.1,.5,.9]], dtype='float32'))
            ae2run2 = ae2.run(np.array([[.1,.5,.9]], dtype='float32'))

            self.assertFalse(np.array_equal(aerun2, aerun1))
            self.assertFalse(np.array_equal(ae2run2, ae2run1))
            self.assertTrue(np.array_equal(aerun2, ae2run2))

            sgd2 = SGD(dataset=dataset, model=ae2, loss=BinaryCrossentropy(inputs=ae2.get_outputs(), targets=x2), epochs=5)
            sgd2.train()

            aerun3 = ae.run(np.array([[.1,.5,.9]], dtype='float32'))
            ae2run3 = ae2.run(np.array([[.1,.5,.9]], dtype='float32'))

            self.assertFalse(np.array_equal(aerun3, aerun2))
            self.assertFalse(np.array_equal(ae2run3, ae2run2))
            self.assertTrue(np.array_equal(aerun3, ae2run3))


        finally:
            del x, e, d, ae, x2, e2, d2, ae2


    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
