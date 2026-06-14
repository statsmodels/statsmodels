"""
Vendored docstring decorators — previously imported from pandas.util._decorators.
No longer depend on pandas internals.

Originally derived from matplotlib.docstring (1.1.0).
Vendored to remove statsmodels' dependency on pandas private API.
"""

from __future__ import annotations

from functools import wraps
from textwrap import dedent
from typing import Any, Callable, Mapping, TypeVar
import warnings

F = TypeVar("F", bound=Callable[..., Any])


def deprecate_kwarg(
    old_arg_name: str,
    new_arg_name: str | None,
    mapping: Mapping[Any, Any] | Callable[[Any], Any] | None = None,
    stacklevel: int = 2,
) -> Callable[[F], F]:
    """
    Decorator to deprecate a keyword argument of a function.

    Vendored from pandas.util._decorators to remove dependency on
    pandas private API.

    Parameters
    ----------
    old_arg_name : str
        Name of argument in function to deprecate.
    new_arg_name : str or None
        Name of preferred argument in function. Use None to raise
        warning that ``old_arg_name`` keyword is deprecated with
        no replacement.
    mapping : dict or callable, optional
        If mapping is present, use it to translate old arguments to
        new arguments. A callable must do its own value checking;
        values not found in a dict will be forwarded unchanged.
    stacklevel : int, default 2
        Stack level for the warning.

    Examples
    --------
    The following deprecates 'cols', using 'columns' instead:

    >>> @deprecate_kwarg(old_arg_name='cols', new_arg_name='columns')
    ... def f(columns=''):
    ...     print(columns)
    """
    if mapping is not None and not hasattr(mapping, "get") and not callable(mapping):
        raise TypeError(
            "mapping from old to new argument values must be dict or callable!"
        )

    def _deprecate_kwarg(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            __tracebackhide__ = True
            old_arg_value = kwargs.pop(old_arg_name, None)

            if old_arg_value is not None:
                if new_arg_name is None:
                    msg = (
                        f"the {old_arg_name!r} keyword is deprecated and "
                        "will be removed in a future version. Please take "
                        f"steps to stop the use of {old_arg_name!r}"
                    )
                    warnings.warn(msg, FutureWarning, stacklevel=stacklevel)
                    kwargs[old_arg_name] = old_arg_value
                    return func(*args, **kwargs)

                elif mapping is not None:
                    if callable(mapping):
                        new_arg_value = mapping(old_arg_value)
                    else:
                        new_arg_value = mapping.get(old_arg_value, old_arg_value)
                    msg = (
                        f"the {old_arg_name}={old_arg_value!r} keyword is "
                        f"deprecated, use "
                        f"{new_arg_name}={new_arg_value!r} instead."
                    )
                else:
                    new_arg_value = old_arg_value
                    msg = (
                        f"the {old_arg_name!r} keyword is deprecated, "
                        f"use {new_arg_name!r} instead."
                    )

                warnings.warn(msg, FutureWarning, stacklevel=stacklevel)
                if kwargs.get(new_arg_name) is not None:
                    msg = (
                        f"Can only specify {old_arg_name!r} "
                        f"or {new_arg_name!r}, not both."
                    )
                    raise TypeError(msg)
                kwargs[new_arg_name] = new_arg_value
            return func(*args, **kwargs)

        return wrapper

    return _deprecate_kwarg


class Appender:
    """
    A function decorator that will append an addendum to the docstring
    of the target function.

    This decorator is robust even if func.__doc__ is None
    (for example, if -OO was passed to the interpreter).

    Usage: construct an Appender with a string to be joined to
    the original docstring. An optional 'join' parameter may be supplied
    which will be used to join the docstring and addendum. e.g.

        add_copyright = Appender("Copyright (c) 2009", join='\n')

        @add_copyright
        def my_dog(has='fleas'):
            "This docstring will have a copyright below"
            pass

    Parameters
    ----------
    addendum : str or None
        String to append to the wrapped function's docstring.
    join : str, optional
        A string placed between the original docstring and the addendum.
        Default is "".
    indents : int, optional
        Number of indents (4-space blocks) added to all lines of the
        addendum. Default is 0.

    """

    addendum: str | None

    def __init__(self, addendum: str | None, join: str = "", indents: int = 0) -> None:
        if indents > 0:
            self.addendum = indent(addendum, indents=indents)
        else:
            self.addendum = addendum
        self.join = join

    def __call__(self, func):
        func.__doc__ = func.__doc__ if func.__doc__ else ""
        self.addendum = self.addendum if self.addendum else ""
        docitems = [func.__doc__, self.addendum]
        func.__doc__ = dedent(self.join.join(docitems))
        return func


class Substitution:
    """
    A decorator to take a function's docstring and perform string
    substitution on it.

    This decorator is robust even if func.__doc__ is None
    (for example, if -OO was passed to the interpreter).

    Usage: construct a Substitution with a sequence or dictionary
    suitable for performing substitution; then decorate a suitable
    function with the constructed object. e.g.

        sub_author_name = Substitution(author='Jason')

        @sub_author_name
        def some_function(x):
            "%(author)s wrote this function"

        # note that some_function.__doc__ is now "Jason wrote this function"

    One can also use positional arguments:

        sub_first_last_names = Substitution('Edgar Allen', 'Poe')

        @sub_first_last_names
        def some_function(x):
            "%s %s wrote the Raven"

    Parameters
    ----------
    *args : str
        Positional arguments for %s-style substitution.
    **kwargs : str
        Keyword arguments for %(name)s-style substitution.
        Cannot be combined with positional args.

    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        if args and kwargs:
            raise AssertionError("Only positional or keyword args are allowed")
        self.params: tuple | dict = args or kwargs

    def __call__(self, func):
        func.__doc__ = func.__doc__ and func.__doc__ % self.params
        return func

    def update(self, *args: object, **kwargs: object) -> None:
        """
        Update self.params with supplied args.

        Only valid when Substitution was constructed with keyword arguments
        (i.e. self.params is a dict). No-op for positional (tuple) params.
        """
        if isinstance(self.params, dict):
            self.params.update(*args, **kwargs)


def indent(text: str | None, indents: int = 1) -> str:
    """
    Add indentation to each line of text.

    Uses 4-space blocks per indent level, matching pandas' original behavior.

    Parameters
    ----------
    text : str or None
        The text to indent. Returns "" if None or empty.
    indents : int, optional
        Number of 4-space indent levels to add. Default is 1.

    Returns
    -------
    str
        Indented text, or "" if input was None/empty.

    """
    if not text or not isinstance(text, str):
        return ""
    jointext = "".join(["\n"] + ["    "] * indents)
    return jointext.join(text.split("\n"))
