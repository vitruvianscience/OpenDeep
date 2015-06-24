'''
A sequential dataset iterator - pull each example from the dataset in the order it is stored.
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
# internal references
from opendeep.data.iterators.iterator import Iterator
import opendeep.data.dataset as datasets

log = logging.getLogger(__name__)

class SequentialIterator(Iterator):
    '''
    An iterator that goes through a dataset in its stored sequence
    '''
    def __init__(self, dataset, unsupervised=False, subset=datasets.TRAIN,
                 batch_size=1, minimum_batch_size=1,
                 rng=None):
        log.debug('Initializing a %s sequential iterator over %s, unsupervised=%s',
                  str(type(dataset)), datasets.get_subset_strings(subset), str(unsupervised))
        super(self.__class__, self).__init__(dataset, unsupervised, subset, batch_size, minimum_batch_size, rng)

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
            # increment the iteration index
            self.iteration_index += 1
            # grab the data and labels to return
            data = self.dataset.getDataByIndices(indices=list(range(start_index, end_index)),
                                                 subset=self.subset)
            # if this is an unsupervised iteration, only return the data (saves some time by not evaluating labels)
            if self.unsupervised:
                return data, None

            labels = self.dataset.getLabelsByIndices(indices=list(range(start_index, end_index)),
                                                     subset=self.subset)
            return data, labels
        else:
            raise StopIteration()