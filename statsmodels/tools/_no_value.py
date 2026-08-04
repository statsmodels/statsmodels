"""
Module defining the singleton sentinel used to mark instance attributes
that have not been set.

Modelled on ``numpy._NoValue`` (``numpy/_globals.py``).  Attributes that
are only computed on some code paths should be initialized to ``_NoValue``
in ``__init__`` and tested with ``attr is _NoValue`` instead of probing the
instance with ``hasattr``/``getattr(..., default)``.  Unlike ``None``,
``_NoValue`` cannot collide with a legitimate attribute value.

This module raises a RuntimeError if an attempt to reload it is made, so
the identity of ``_NoValue`` is fixed even if statsmodels is reloaded and
``is`` comparisons remain valid.

See GH#9880.
"""

__all__ = ["_NoValue"]

# Disallow reloading this module so as to preserve the identity of _NoValue.
if "_is_loaded" in globals():
    raise RuntimeError("Reloading statsmodels.tools._no_value is not allowed")
_is_loaded = True


class _NoValueType:
    """Special value indicating an attribute has not been set.

    Use the ``_NoValue`` singleton instance of this class, and test with
    ``is``::

        self.attr = _NoValue
        ...
        if self.attr is not _NoValue:
            ...

    Do not rely on truthiness; ``_NoValue`` is truthy like any plain
    object.
    """

    __instance = None

    def __new__(cls):
        # ensure that only one instance exists
        if not cls.__instance:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __reduce__(self):
        # preserve singleton identity across pickle round-trips for every
        # protocol, so `is` checks hold on unpickled results objects
        return (self.__class__, ())

    def __repr__(self):
        return "<no value>"


_NoValue = _NoValueType()
