"""
.. module:: iterator

General interface for creating a dataset iterator object.
"""
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import logging
# third party libraries
import numpy
# internal references
import opendeep.data.dataset as datasets

log = logging.getLogger(__name__)

# variables for the dataset iteration modes
SEQUENTIAL = 0
RANDOM     = 1

class Iterator(object):
    '''
    Default interface for a Dataset iterator
    '''
    def __init__(self, dataset=None, subset=None, batch_size=1, minimum_batch_size=1, rng=None):
        # make sure the subset is recognized
        if subset not in [datasets.TRAIN, datasets.VALID, datasets.TEST]:
            log.error('Dataset subset %s not recognized, try TRAIN, VALID, or TEST', datasets.get_subset_strings(subset))
        self.dataset = dataset
        self.subset = subset
        self.batch_size = batch_size
        self.minimum_batch_size = minimum_batch_size

        # determine the number of possible iterations given the batch size, minimum batch size, dataset, and subset
        self.data_len = self.dataset.getDataShape(self.subset)[0]
        batches = self.data_len/self.batch_size
        self.iterations = batches*[batch_size]

        remainder = numpy.remainder(self.data_len, self.batch_size)
        if remainder >= self.minimum_batch_size:
            self.iterations.append(remainder)

        self.iteration_index = 0

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


