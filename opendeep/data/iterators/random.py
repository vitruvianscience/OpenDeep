'''
A random dataset iterator
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

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
    def __init__(self, dataset, subset=datasets.TRAIN, batch_size=1, minimum_batch_size=1, rng=None):
        # initialize a numpy rng if one is not provided
        if rng is None:
            random.seed(123)
            self.rng = random
        else:
            self.rng = rng

        _t = time.time()
        log.debug('Initializing a %s random iterator over %s', str(type(dataset)), datasets.get_subset_strings(subset))
        super(self.__class__, self).__init__(dataset, subset, batch_size, minimum_batch_size)

        # randomize the indices to access
        self.indices = numpy.arange(self.data_len)
        self.rng.shuffle(self.indices)
        log.debug('iterator took %s to make' % make_time_units_string(time.time() - _t))

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
            # convert the iteration index into the start and end indices for the batch in the dataset
            _start_index = self.iteration_index*self.batch_size
            _end_index   = _start_index + self.iterations[self.iteration_index]
            indices_this_step = self.indices[_start_index:_end_index]
            # increment the iteration index
            self.iteration_index += 1
            # grab the labels and data to return
            data = self.dataset.getDataByIndices(indices=indices_this_step,
                                                 subset=self.subset)
            labels = self.dataset.getLabelsByIndices(indices=indices_this_step,
                                                     subset=self.subset)

            return data, labels
        else:
            raise StopIteration()