import theano.tensor as T
from opendeep.models.model import Model
from opendeep.utils.nnet import get_weights_uniform, get_bias
from opendeep.utils.noise import salt_and_pepper
from opendeep.utils.activation import tanh, sigmoid
from opendeep.utils.cost import binary_crossentropy

# create our class initialization!
class DenoisingAutoencoder(Model):
    """
    A denoising autoencoder will corrupt an input (add noise) and try to reconstruct it.
    """
    def __init__(self):
        # Define some model hyperparameters to work with MNIST images!
        input_size  = 28*28 # dimensions of image
        hidden_size = 1000  # number of hidden units - generally bigger than input size for DAE

        # Now, define the symbolic input to the model (Theano)
        # We use a matrix rather than a vector so that minibatch processing can be done in parallel.
        x = T.matrix("X")
        self.inputs = [x]

        # Build the model's parameters - a weight matrix and two bias vectors
        W  = get_weights_uniform(shape=(input_size, hidden_size), name="W")
        b0 = get_bias(shape=input_size, name="b0")
        b1 = get_bias(shape=hidden_size, name="b1")
        self.params = [W, b0, b1]

        # Perform the computation for a denoising autoencoder!
        # first, add noise (corrupt) the input
        corrupted_input = salt_and_pepper(input=x, noise_level=0.4)
        # next, run the hidden layer given the inputs (the encoding function)
        hiddens = tanh(T.dot(corrupted_input, W) + b1)
        # finally, create the reconstruction from the hidden layer (we tie the weights with W.T)
        reconstruction = sigmoid(T.dot(hiddens, W.T) + b0)
        # the training cost is reconstruction error - with MNIST this is binary cross-entropy
        self.train_cost = binary_crossentropy(output=reconstruction, target=x)

        # Compile everything into a Theano function for prediction!
        # When using real-world data in predictions, we wouldn't corrupt the input first.
        # Therefore, create another version of the hiddens and reconstruction without adding the noise
        hiddens_predict      = tanh(T.dot(x, W) + b1)
        self.recon_predict   = sigmoid(T.dot(hiddens_predict, W.T) + b0)

    def get_inputs(self):
        return self.inputs

    def get_params(self):
        return self.params

    def get_train_cost(self):
        return self.train_cost

    def get_outputs(self):
        return self.recon_predict

if __name__ == '__main__':
    # set up the logging environment to display outputs (optional)
    # although this is recommended over print statements everywhere
    import logging
    from opendeep.log.logger import config_root_logger
    config_root_logger()
    log = logging.getLogger(__name__)
    log.info("Creating a Denoising Autoencoder!")

    # import the dataset and optimizer to use
    from opendeep.data.dataset import TEST
    from opendeep.data.standard_datasets.image.mnist import MNIST
    from opendeep.optimization.adadelta import AdaDelta

    # grab the MNIST dataset
    mnist = MNIST()

    # create your shiny new DAE
    dae = DenoisingAutoencoder()

    # make an optimizer to train it (AdaDelta is a good default)
    optimizer = AdaDelta(model=dae, dataset=mnist)
    # perform training!
    optimizer.train()

    # test it on some images!
    test_data, _ = mnist.getSubset(TEST)
    test_data = test_data[:25].eval()
    corrupted_test = salt_and_pepper(test_data, 0.4).eval()
    # use the run function!
    reconstructed_images = dae.run(corrupted_test)

    # create an image from this reconstruction!
    # imports for working with tiling outputs into one image
    from opendeep.utils.image import tile_raster_images
    import numpy
    import PIL
    # stack the image matrices together in three 5x5 grids next to each other using numpy
    stacked = numpy.vstack(
        [numpy.vstack([test_data[i*5 : (i+1)*5],
                       corrupted_test[i*5 : (i+1)*5],
                       reconstructed_images[i*5 : (i+1)*5]])
         for i in range(5)])
    # convert the combined matrix into an image
    image = PIL.Image.fromarray(
        tile_raster_images(stacked, (28, 28), (5, 3*5))
    )
    # save it!
    image.save("dae_reconstruction_test.png")