"""
Making an RNN-GSN using separate models in a Prototype.
"""

# standard libraries
import math
# third party
import theano
# internal references
from opendeep.data.standard_datasets.image.mnist import MNIST
from opendeep.log.logger import config_root_logger
from opendeep.models.model import Model
from opendeep.models.multi_layer.recurrent import RNN
from opendeep.models.multi_layer.generative_stochastic_network import GSN
from opendeep.optimization.rmsprop import RMSProp

class RNN_GSN(Model):
    def __init__(self):
        super(RNN_GSN, self).__init__()

        gsn_hiddens = 500
        gsn_layers = 2

        # RNN that takes in images (3D sequences) and outputs gsn hiddens (3D sequence of them)
        self.rnn = RNN(
            input_size=28 * 28,
            hidden_size=100,
            # needs to output hidden units for odd layers of GSN
            output_size=gsn_hiddens * (math.ceil(gsn_layers/2.)),
            layers=1,
            activation='tanh',
            hidden_activation='relu',
            weights_init='uniform', weights_interval='montreal',
            r_weights_init='identity'
        )

        # Create the GSN that will encode the input space
        gsn = GSN(
            input_size=28 * 28,
            hidden_size=gsn_hiddens,
            layers=gsn_layers,
            walkbacks=4,
            visible_activation='sigmoid',
            hidden_activation='tanh',
            image_height=28,
            image_width=28
        )
        # grab the input arguments
        gsn_args = gsn.args.copy()
        # grab the parameters it initialized
        gsn_params = gsn.get_params()

        # Now hook the two up! RNN should output hiddens for GSN into a 3D tensor (1 set for each timestep)
        # Therefore, we need to use scan to create the GSN reconstruction for each timestep given the hiddens
        def step(hiddens, x):
            gsn = GSN(
                inputs_hook=(28*28, x),
                hiddens_hook=(gsn_hiddens, hiddens),
                params_hook=(gsn_params),
                **gsn_args
            )
            # return the reconstruction and cost!
            return gsn.get_outputs(), gsn.get_train_cost()

        (outputs, costs), scan_updates = theano.scan(
            fn=lambda h, x: step(h, x),
            sequences=[self.rnn.output, self.rnn.input],
            outputs_info=[None, None]
        )

        self.outputs = outputs

        self.updates = dict()
        self.updates.update(self.rnn.get_updates())
        self.updates.update(scan_updates)

        self.cost = costs.sum()
        self.params = gsn_params + self.rnn.get_params()


    def get_inputs(self):
        return self.rnn.get_inputs()
    def get_params(self):
        return self.params
    def get_train_cost(self):
        return self.cost
    def get_updates(self):
        return self.updates
    def get_outputs(self):
        return self.outputs


def main(sequence):
    rnn_gsn = RNN_GSN()

    # data! (needs to be 3d for rnn).
    mnist = MNIST(sequence_number=sequence, seq_3d=True, seq_length=50)

    # optimizer!
    optimizer = RMSProp(
        model=rnn_gsn,
        dataset=mnist,
        epochs=500,
        batch_size=50,
        save_freq=10,
        stop_patience=30,
        stop_threshold=.9995,
        learning_rate=1e-6,
        decay=.95,
        max_scaling=1e5,
        grad_clip=5.,
        hard_clip=False
    )
    # train!
    optimizer.train()


if __name__ == "__main__":
    config_root_logger()
    sequence = 1
    main(sequence)