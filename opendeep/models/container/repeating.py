"""
This module defines a container for quickly assembling layers/models repeating over time
together without needing to define a new :class:`Model` class. This should mainly be used
for experimentation, and then later you should make your creation into a new :class:`Model` class.
"""
# standard libraries
import logging
# third party libraries
from theano.tensor import TensorType
# internal references
from opendeep.models.model import Model
from opendeep.utils.misc import raise_to_list

log = logging.getLogger(__name__)


class Repeating(Model):
    """
    The `Repeating` container takes a `Model` instance and repeats it across the first dimension of the input.
    """
    def __init__(self, model):
        raise NotImplementedError("Repeating class not implemented yet!")
        # make sure the input model to repeat is a Model instance
        assert isinstance(model, Model), "The initial model provided was type %s, not a Model." % str(type(model))
        self.model = model
        # make this input one dimension more than the provided Model's input (since we are repeating over the
        # first dimension)
        model_input = raise_to_list(self.model.get_inputs())[0]
        self.input = TensorType(model_input.dtype, (False,)*(model_input.ndim + 1))
