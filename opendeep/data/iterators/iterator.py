"""
General interface for creating a dataset iterator object.
"""
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import logging
# third party libraries
import numpy
# internal references
import opendeep.data.dataset as datasets
from opendeep.utils.misc import raise_to_list

log = logging.getLogger(__name__)

# variables for the dataset iteration modes
SEQUENTIAL = 0
RANDOM     = 1

class Iterator(object):
    '''
    Default interface for a Dataset iterator
    '''
    def __init__(self, dataset=None, unsupervised=False, subset=None, batch_size=1, minimum_batch_size=1, rng=None):
        # make sure the subset is recognized
        if subset not in [datasets.TRAIN, datasets.VALID, datasets.TEST]:
            log.error('Dataset subset %s not recognized, try TRAIN, VALID, or TEST',
                      datasets.get_subset_strings(subset))
        self.dataset = dataset
        self.unsupervised = unsupervised
        self.subset = subset
        self.batch_size = int(batch_size)
        self.minimum_batch_size = int(minimum_batch_size)

        # determine the number of possible iterations given the batch size, minimum batch size, dataset, and subset
        data_shapes = raise_to_list(self.dataset.getDataShape(self.subset))
        data_lens = [shape[0] for shape in data_shapes]

        # self.iterations will hold the list of tuples (start, end) for grabbing segments of the dataset.
        self.iterations = []
        start_idx = 0
        for data_len in data_lens:
            # integer division to determine number of whole batches for this length
            n_batches = data_len/int(self.batch_size)
            # add the (start_idx, end_idx) tuple to the self.iterations list
            for i in range(n_batches):
                end_idx = start_idx + self.batch_size
                self.iterations.append((start_idx, end_idx))
                start_idx = end_idx
            # remainder to find number of leftover examples
            remainder = numpy.remainder(data_len, self.batch_size)
            end_idx = start_idx + remainder
            # check if it is bigger than the minimum allowed size
            if remainder >= self.minimum_batch_size:
                self.iterations.append((start_idx, end_idx))
            start_idx = end_idx

        self.iteration_index = 0
        self.total_data_len = numpy.sum(data_lens)

    def __iter__(self):
        return self

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
        log.critical('Iterator %s doesn\'t have a next() method', str(type(self)))
        raise NotImplementedError()


