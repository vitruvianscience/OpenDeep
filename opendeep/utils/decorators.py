"""
This module provides various decorators used throughout OpenDeep.
"""
from functools import wraps
# __all__ is created at the bottom of this file - go there for publicly available decorator names.

class DocInherit(object):
    """
    Docstring inheriting method descriptor. This is used for copying docstrings from a superclass to subclass
    implementations. Taken from
    http://stackoverflow.com/questions/2025562/inherit-docstrings-in-python-class-inheritance

    The class itself is also used as a decorator.
    """
    def __init__(self, mthd):
        self.mthd = mthd
        self.name = mthd.__name__

    def __get__(self, obj, cls):
        if obj:
            return self.get_with_inst(obj, cls)
        else:
            return self.get_no_inst(cls)

    def get_with_inst(self, obj, cls):

        overridden = getattr(super(cls, obj), self.name, None)

        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            return self.mthd(obj, *args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def get_no_inst(self, cls):

        for parent in cls.__mro__[1:]:
            overridden = getattr(parent, self.name, None)
            if overridden: break

        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            return self.mthd(*args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def use_parent_doc(self, func, source):
        if source is None:
            raise NameError("Can't find '%s' in parents" % self.name)
        func.__doc__ = source.__doc__
        return func

#########################################################
# List all of the wrapper names here
#########################################################
doc = DocInherit

__all__ = ['doc']