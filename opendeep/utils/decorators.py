"""
This module provides various decorators used throughout OpenDeep.
"""
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

#########################################################
# List all of the wrapper names here
#########################################################
inherit_docs = inherit_missing_function_docs

__all__ = ['inherit_docs']