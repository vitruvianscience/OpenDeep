"""
This module provides various decorators used throughout OpenDeep.
"""
from opendeep.optimization.optimizer import Optimizer
from opendeep.models.model import Model

# __all__ is created at the bottom of this file - go there for publicly available decorator names.


def inherit_missing_function_docs(cls):
    """
    Goes through all functions in the class and replaces empty docstrings with the superclass's docstring.
    """
    for name, func in vars(cls).items():
        if not func.__doc__:
            for parent in cls.__bases__:
                try:
                    parfunc = getattr(parent, name)
                    if parfunc and getattr(parfunc, '__doc__', None):
                        func.__doc__ = parfunc.__doc__
                        break
                except AttributeError:
                    pass
    return cls

def init_optimizer(train_method):
    """
    Takes the optimizer given and initializes it with the model.
    """
    def wrapper(*args, **kwargs):
        # find the optimizer that was passed to the train function - either kwargs or the very first argument.
        optimizer = kwargs.pop("optimizer", None)
        if not optimizer:
            optimizer = args[1]
        # get the initialization parameters from the optimizer
        assert isinstance(optimizer, Optimizer), "Input optimizer needs to be an Optimizer! Found %s" % \
            str(type(optimizer))
        init_params = optimizer.args.copy()
        # add the model's 'self' (must be args[0] of the method) to the optimizer initial config.
        model = args[0]
        assert isinstance(model, Model), "First argument (normally 'self') to the method must be a Model! " \
                                         "Found %s" % str(type(model))
        init_params['model'] = model
        new_optimizer = type(optimizer)(**init_params)
        return train_method(new_optimizer)
    return wrapper


#########################################################
# List all of the wrapper names here
#########################################################
inherit_docs = inherit_missing_function_docs

__all__ = ['inherit_docs', 'init_optimizer']
