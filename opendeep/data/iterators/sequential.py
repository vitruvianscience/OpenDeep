'''
A sequential dataset iterator
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
# internal references
from opendeep.data.iterators.iterator import Iterator
import opendeep.data.dataset as datasets
from opendeep.utils.misc import make_time_units_string

log = logging.getLogger(__name__)

class SequentialIterator(Iterator):
    '''
    An iterator that goes through a dataset in its stored sequence
    '''
    def __init__(self, dataset, subset=datasets.TRAIN, batch_size=1, minimum_batch_size=1, rng=None):
        _t = time.time()
        log.debug('Initializing a %s sequential iterator over %s', str(type(dataset)), datasets.get_subset_strings(subset))
        super(self.__class__, self).__init__(dataset, subset, batch_size, minimum_batch_size, rng)
        log.debug('iterator took %s to make' % make_time_units_string(time.time()-_t))

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
            # increment the iteration index
            self.iteration_index += 1
            # grab the data and labels to return
            data = self.dataset.getDataByIndices(indices=list(range(_start_index, _end_index)),
                                                 subset=self.subset)
            labels = self.dataset.getLabelsByIndices(indices=list(range(_start_index, _end_index)),
                                                     subset=self.subset)

            return data, labels
        else:
            raise StopIteration()