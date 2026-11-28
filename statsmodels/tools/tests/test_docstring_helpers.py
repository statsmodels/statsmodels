import pytest

from statsmodels.tools.docstring_helpers import Appender, Substitution, indent


@pytest.mark.parametrize(
    "indents, expected",
    [
        (0, "Doc.\nAdded\nlines"),
        (1, "Doc.\nAdded\n    lines"),
        (2, "Doc.\nAdded\n        lines"),
    ],
    ids=["0", "1", "2"],
)
def test_appender_indents(indents, expected):
    # Each indent adds 4 spaces to the lines of the addendum after the first
    @Appender("Added\nlines", join="\n", indents=indents)
    def f():
        "Doc."

    assert f.__doc__ == expected


def test_appender_none():
    # A docstring or an addendum of None (as under python -OO) counts as ""
    def f():
        pass

    f.__doc__ = None
    assert Appender("Added", join="\n")(f).__doc__ == "\nAdded"

    def g():
        "Doc."

    assert Appender(None, join="\n")(g).__doc__ == "Doc.\n"


def test_substitution_update_dict_params():
    # update() mutates the dict of keyword substitution params in place,
    # so a later application of the decorator picks up the new value
    sub = Substitution(name="Ada")

    @sub
    def f():
        "Hello %(name)s"

    assert f.__doc__ == "Hello Ada"

    sub.update(name="Grace")

    @sub
    def g():
        "Hello %(name)s"

    assert g.__doc__ == "Hello Grace"


def test_substitution_update_noop_for_positional_params():
    # update() is documented as a no-op when Substitution was constructed
    # with positional args (self.params is a tuple, not a dict)
    sub = Substitution("Ada", "Lovelace")

    sub.update(extra="ignored")

    assert sub.params == ("Ada", "Lovelace")

    @sub
    def f():
        "%s %s wrote this"

    assert f.__doc__ == "Ada Lovelace wrote this"


def test_indent_multiline():
    # first line is left alone; every subsequent line gets `indents` many
    # 4-space blocks prepended
    assert indent("a\nb\nc", indents=1) == "a\n    b\n    c"
    assert indent("a\nb", indents=2) == "a\n        b"
    assert indent("a\nb", indents=0) == "a\nb"


def test_indent_single_line_unchanged():
    assert indent("solo", indents=3) == "solo"


def test_indent_none_or_empty_returns_empty_string():
    assert indent(None) == ""
    assert indent("") == ""


def test_indent_non_string_returns_empty_string():
    # documented as robust to non-str/None input, returning ""
    assert indent(12345) == ""
