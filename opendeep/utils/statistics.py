"""
This module is used for computing statistics such as mean, variance, mode, regularization values, etc.
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
import theano.tensor as T
import theano.compat.six as six
# internal imports
from opendeep.utils.misc import raise_to_list

log = logging.getLogger(__name__)

def get_stats(input, stat=None):
    """
    Returns a dictionary mapping the name of the statistic to the result on the input.
    Currently gets mean, var, std, min, max, l1, l2.

    Parameters
    ----------
    input : tensor
        Theano tensor to grab stats for.

    Returns
    -------
    dict
        Dictionary of all the statistics expressions {string_name: theano expression}
    """
    stats = {
        'mean': T.mean(input),
        'var': T.var(input),
        'std': T.std(input),
        'min': T.min(input),
        'max': T.max(input),
        'l1': input.norm(L=1),
        'l2': input.norm(L=2),
        #'num_nonzero': T.sum(T.nonzero(input)),
    }
    stat_list = raise_to_list(stat)
    compiled_stats = {}
    if stat_list is None:
        return stats

    for stat in stat_list:
        if isinstance(stat, six.string_types) and stat in stats:
            compiled_stats.update({stat: stats[stat]})
    return compiled_stats