from __future__ import print_function
# third party libraries
from theano.tensor import matrix, lvector
from opendeep.log.logger import config_root_logger
from opendeep.models import Prototype, Dense, Softmax
from opendeep.optimization import AdaDelta
from opendeep.optimization.loss import Neg_LL
from opendeep.data import MNIST

def run_mlp():
    # test the new way to automatically fill in inputs for models
    mlp = Prototype()
    x = ((None, 784), matrix("x"))
    mlp.add(Dense(inputs=x, outputs=1000, activation='rectifier', noise='dropout'))
    mlp.add(Dense, outputs=1500, activation='tanh', noise='dropout')
    mlp.add(Softmax, outputs=10, out_as_probs=False)

    # define our loss to optimize for the model (and the target variable)
    # targets from MNIST are int64 numbers 0-9
    y = lvector('y')
    loss = Neg_LL(inputs=mlp.models[-1].p_y_given_x, targets=y, one_hot=False)

    mnist = MNIST()

    optimizer = AdaDelta(model=mlp, loss=loss, dataset=mnist, epochs=10)
    optimizer.train()

    test_data, test_labels = mnist.test_inputs, mnist.test_targets
    test_data = test_data[:25]
    test_labels = test_labels[:25]
    # use the run function!
    yhat = mlp.run(test_data)
    print('-------')
    print('Prediction: %s' % str(yhat))
    print('Actual:     %s' % str(test_labels.astype('int32')))



if __name__ == '__main__':
    config_root_logger()
    run_mlp()
