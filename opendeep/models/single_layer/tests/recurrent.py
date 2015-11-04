# third party
from theano.tensor import itensor3
# internal imports
from opendeep.log.logger import config_root_logger
from opendeep.models import GRU, LSTM, RNN

def main():
    xs = itensor3('xs')
    ins = ((None, None, 93), xs)
    gru = GRU(
        inputs=ins,
        hiddens=128,
        direction='bidirectional'
    )
    print("GRU output (hiddens) shape: ", gru.output_size)
    print("GRU params: ", gru.get_params())

    lstm = LSTM(
        inputs=ins,
        hiddens=128,
        direction='bidirectional'
    )
    print("LSTM output (hiddens) shape: ", lstm.output_size)
    print("LSTM params: ", lstm.get_params())

    rnn = RNN(
        inputs=ins,
        hiddens=128,
        direction='bidirectional'
    )
    print("RNN output (hiddens) shape: ", rnn.output_size)
    print("RNN params: ", rnn.get_params())

if __name__ == '__main__':
    config_root_logger()
    main()
