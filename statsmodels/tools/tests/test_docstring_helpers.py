from statsmodels.tools.docstring_helpers import Substitution, indent


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
