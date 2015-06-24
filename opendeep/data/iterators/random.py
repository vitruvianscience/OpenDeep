'''
.. module:: random

A random dataset iterator - pull each example from the dataset in a random order.
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
import time
# third party libraries
import numpy
import numpy.random as random
# internal references
from opendeep.data.iterators.iterator import Iterator
import opendeep.data.dataset as datasets
from opendeep.utils.misc import make_time_units_string

log = logging.getLogger(__name__)

class RandomIterator(Iterator):
    '''
    An iterator that goes through a dataset in a random sequence
    '''
    def __init__(self, dataset, unsupervised=False, subset=datasets.TRAIN,
                 batch_size=1, minimum_batch_size=1,
                 rng=None):
        # initialize a numpy rng if one is not provided
        if rng is None:
            random.seed(123)
            self.rng = random
        else:
            self.rng = rng

        log.debug('Initializing a %s random iterator over %s, unsupervised=%s',
                  str(type(dataset)), datasets.get_subset_strings(subset), str(unsupervised))
        super(self.__class__, self).__init__(dataset, unsupervised, subset, batch_size, minimum_batch_size)

        # randomize the indices to access
        self.indices = numpy.arange(self.total_data_len)
        self.rng.shuffle(self.indices)

    def next(self):
        '''
        Gets the next examples(s) based on the batch size
        :return: tuple
        Batch of data values and labels from the dataset

        :raises: StopIteration
        When there are no more batches that meet the minimum requirement to return

        The intention of the protocol is that once an iterator's next() method raises StopIteration, it will continue
        to do so on subsequent calls. Implementations that do not obey this property are deemed broken.
        '''
        if self.iteration_index < len(self.iterations):
            # grab the start and end indices for the batch in the dataset
            start_index, end_index = self.iterations[self.iteration_index]
            indices_this_step = self.indices[start_index:end_index]
            # increment the iteration index
            self.iteration_index += 1
            # grab the labels and data to return
            data = self.dataset.getDataByIndices(indices=indices_this_step,
                                                 subset=self.subset)
            # if this is an unsupervised iteration, only return the data (saves some time by not evaluating labels)
            if self.unsupervised:
                return data, None
            labels = self.dataset.getLabelsByIndices(indices=indices_this_step,
                                                     subset=self.subset)
            return data, labels
        else:
            raise StopIteration()