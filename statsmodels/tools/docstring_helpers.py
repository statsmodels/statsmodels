"""
Vendored docstring decorators

Previously imported from pandas.util._decorators. No longer depend on
pandas internals.

Originally derived from matplotlib.docstring (1.1.0). Vendored to remove
statsmodels' dependency on pandas private API.
"""

from __future__ import annotations

from textwrap import dedent


class Appender:
    """
    A function decorator that will append an addendum to the docstring
    of the target function

    This decorator is robust even if func.__doc__ is None
    (for example, if -OO was passed to the interpreter).


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

    Examples
    --------
    Usage: construct an Appender with a string to be joined to
    the original docstring. An optional 'join' parameter may be supplied
    which will be used to join the docstring and addendum. e.g.

    >>> add_copyright = Appender("Copyright (c) 2009", join=" ")

    >>> @add_copyright
    ... def my_dog(has='fleas'):
    ...     "This docstring will have a copyright notice appended to it."
    ...     pass
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
    substitution on it

    This decorator is robust even if func.__doc__ is None
    (for example, if -OO was passed to the interpreter).

    Parameters
    ----------
    *args : str
        Positional arguments for %s-style substitution.
    **kwargs : str
        Keyword arguments for %(name)s-style substitution.
        Cannot be combined with positional args.

    Examples
    --------
    Usage: construct a Substitution with a sequence or dictionary
    suitable for performing substitution; then decorate a suitable
    function with the constructed object. e.g.

    >>> sub_author_name = Substitution(author='Jason')

    >>> @sub_author_name
    >>> def some_function(x):
    ...    "%(author)s wrote this function"

    Note that some_function.__doc__ is now "Jason wrote this function"

    One can also use positional arguments:

    >>> sub_first_last_names = Substitution('Edgar Allen', 'Poe')

    >>> @sub_first_last_names
    >>> def some_function(x):
    ...    "%s %s wrote the Raven"
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
        Update self.params with supplied args

        Only valid when Substitution was constructed with keyword arguments
        (i.e. self.params is a dict). No-op for positional (tuple) params.
        """
        if isinstance(self.params, dict):
            self.params.update(*args, **kwargs)


def indent(text: str | None, indents: int = 1) -> str:
    """
    Add indentation to each line of text

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
