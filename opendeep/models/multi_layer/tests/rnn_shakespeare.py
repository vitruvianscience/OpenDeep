# standard libraries
import logging
# third party
import theano.sandbox.rng_mrg as RNG_MRG
# internal imports
from opendeep.log.logger import config_root_logger
from opendeep.data.standard_datasets.text.characters import CharsLM
from opendeep.models.multi_layer.recurrent import RNN
from opendeep.optimization.stochastic_gradient_descent import SGD
from opendeep.optimization.adadelta import AdaDelta
from opendeep.optimization.rmsprop import RMSProp
from opendeep.monitor.monitor import Monitor

log = logging.getLogger(__name__)

def main():
    data = CharsLM(filename='shakespeare_input.txt',
                   source="http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt",
                   seq_length=500, train_split=0.95, valid_split=0.05)

    rnn = RNN(outdir='outputs/rnn/',
              input_size=data.vocab_size,
              hidden_size=100,
              output_size=data.vocab_size,
              layers=3,
              activation='softmax',
              hidden_activation='relu',
              mrg=RNG_MRG.MRG_RandomStreams(1),
              weights_init='uniform',
              weights_interval='montreal',
              bias_init=0.0,
              r_weights_init='identity',
              r_bias_init=0.0,
              cost_function='nll',
              cost_args=None,
              noise='dropout',
              noise_level=.7,
              noise_decay='exponential',
              noise_decay_amount=.99,
              direction='forward')

    cost_monitor = Monitor("cost", rnn.get_train_cost(), train=False, valid=True, test=True)

    optimizer = RMSProp(model=rnn, dataset=data,
                        grad_clip=5., hard_clip=False,
                        learning_rate=2e-3, decay=0.95, batch_size=100, epochs=300)
    # optimizer = AdaDelta(model=gsn, dataset=mnist, n_epoch=200, batch_size=100, learning_rate=1e-6)
    optimizer.train(monitor_channels=cost_monitor)



if __name__ == '__main__':
    config_root_logger()
    main()