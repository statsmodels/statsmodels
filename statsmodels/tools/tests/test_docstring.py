import re

import pytest

from statsmodels.tools.docstring import (
    Docstring,
    NumpyDocString,
    Parameter,
    Reader,
    indent,
    remove_parameters,
)
from statsmodels.tools.sm_exceptions import ParseError

good = """
This is the summary.

This is the extended summary.

Parameters
----------
x : int
    The first parameter.
y : float
    The second parameter.
z : {int, float, None}
    The final parameter.

Returns
-------
float
    Some floating point value.

See Also
--------
statsmodels.api
    The main API location.

Notes
-----
This is where the notes go.

.. index: default
   :refguide: something, else, and more

References
----------
.. [*] Reference 1 here

Examples
--------
Using the API is simple

>>> import statsmodels.api
"""

bad = """
Returns
-------
float
    Some floating point value.

Unknown
-------
I don't know what this section does.
"""

repeat = """
Returns
-------
float
    Some floating point value.

Returns
-------
float
    Some floating point value.
"""

bad_yields = """
Returns
-------
float
    The return.

Yields
------
float
    Can't also yield.
"""

with_sig = """
func(x)

func(x, y)

func(x, y, z=1)
""" + good

bad_receives = """
Receives
--------
x : int
    Can't receive without yielding.
"""


@pytest.mark.parametrize("data", ["a\n\nb", ["a", "", "b"]], ids=["str", "list"])
def test_reader(data):
    # A str is split at the newlines, a list is used as the lines
    reader = Reader(data)
    assert reader[2] == "b"
    assert reader.peek(2) == "b"
    assert [reader.read() for _ in range(3)] == ["a", "", "b"]
    assert reader.eof()
    # Reading or peeking past the last line gives an empty line
    assert reader.read() == ""
    assert reader.peek() == ""
    reader.reset()
    assert reader.read() == "a"


@pytest.mark.parametrize(
    "data, expected", [(" \n\t\n", True), (["", "x"], False)], ids=["blank", "text"]
)
def test_reader_is_empty(data, expected):
    assert Reader(data).is_empty() is expected


def test_numpydocstring_mapping():
    # The mapping has one key per numpydoc section, plus the signature and
    # the index
    sections = {
        "Signature",
        "Summary",
        "Extended Summary",
        "Parameters",
        "Returns",
        "Yields",
        "Receives",
        "Raises",
        "Warns",
        "Other Parameters",
        "Attributes",
        "Methods",
        "See Also",
        "Notes",
        "Warnings",
        "References",
        "Examples",
        "index",
    }
    nds = NumpyDocString(good)
    assert len(nds) == len(sections)
    assert set(nds) == sections
    assert nds["Summary"] == ["This is the summary."]
    assert nds["Returns"] == [Parameter("", "float", ["Some floating point value."])]


def test_remove_parameter():
    ds = Docstring(good)
    ds.remove_parameters("x")
    assert "x : int" not in str(ds)

    ds = Docstring(good)
    ds.remove_parameters(["x", "y"])
    assert "x : int" not in str(ds)
    assert "y : float" not in str(ds)

    with pytest.raises(ValueError):
        Docstring(good).remove_parameters(["w"])

    ds = remove_parameters(good, "x")
    assert "x : int" not in ds
    assert isinstance(ds, str)

    # Docstrings are None under python -OO
    assert remove_parameters(None, "x") is None


@pytest.mark.parametrize(
    "text, predicate, expected",
    [
        (None, None, ""),
        ("a\n\nb", None, "> a\n\n> b"),
        ("a\n\nb", lambda line: True, "> a\n> \n> b"),
    ],
    ids=["none", "default-predicate", "all-lines"],
)
def test_indent(text, predicate, expected):
    # By default only the lines that are not blank are indented
    assert indent(text, "> ", predicate=predicate) == expected


def test_insert_parameters():
    new = Parameter("w", "ndarray", ["An array input."])
    ds = Docstring(good)
    ds.insert_parameters("y", new)
    assert "w : ndarray" in str(ds)
    assert "An array input." in str(ds)

    other = Parameter("q", "DataFrame", ["A pandas dataframe."])
    ds = Docstring(good)
    ds.insert_parameters(None, [new, other])
    assert "w : ndarray" in str(ds)
    assert "An array input." in str(ds)
    assert "q : DataFrame" in str(ds)
    assert "A pandas dataframe." in str(ds)
    assert "---\nw : ndarray" in str(ds)

    ds = Docstring(good)
    with pytest.raises(ValueError):
        ds.insert_parameters("unknown", new)


@pytest.mark.parametrize("desc", [[], [""]], ids=["empty", "blank"])
def test_insert_parameter_without_description(desc):
    # A parameter with an empty or blank description is written as one line
    ds = Docstring(good)
    ds.insert_parameters("x", Parameter("w", "ndarray", desc))
    expected = "w : ndarray\nx : int\n    The first parameter."
    assert ds.extract_parameters(["w", "x"]) == expected


@pytest.mark.parametrize(
    "parameters, indent, expected",
    [
        ("x", 0, "x : int\n    The first parameter."),
        (
            ["z", "x"],
            4,
            (
                "    z : {int, float, None}\n"
                "        The final parameter.\n"
                "    x : int\n"
                "        The first parameter."
            ),
        ),
    ],
    ids=["str", "list-indented"],
)
def test_extract_parameters(parameters, indent, expected):
    # The parameters come out in the requested order, without the header
    ds = Docstring(good)
    assert ds.extract_parameters(parameters, indent=indent) == expected


def test_extract_parameters_missing():
    with pytest.raises(ValueError, match=r"^w were not found in the docstring$"):
        Docstring(good).extract_parameters(["x", "w"])


def test_set_unknown():
    ds = Docstring(good)
    with pytest.raises(ValueError):
        ds._ds["Unknown"] = ["unknown"]


def test_replace_block():
    ds = Docstring(good)
    ds.replace_block("summary", ["The is the new summary."])
    assert "The is the new summary." in str(ds)

    ds = Docstring(good)
    ds.replace_block("summary", "The is the new summary.")
    assert "The is the new summary." in str(ds)

    with pytest.raises(ValueError):
        ds.replace_block("unknown", ["The is the new summary."])


def test_repeat():
    with pytest.raises(ValueError):
        Docstring(repeat)


def test_bad():
    with pytest.raises(ValueError):
        Docstring(bad)


def test_blank_line_before_description():
    # A blank line between a parameter and its description is dropped, and a
    # parameter without a description is written without one
    ds = Docstring("""
Summary.

Parameters
----------
x : int

    The description.
y : float
""")
    expected = """
Summary.

Parameters
----------
x : int
    The description.
y : float
"""
    assert str(ds) == expected


@pytest.mark.parametrize(
    "section", ["Parameters", "Returns", "See Also", "Notes", "Examples"]
)
def test_empty_section(section):
    # A section with a header and no content is dropped
    ds = Docstring(f"Summary.\n\n{section}\n{'-' * len(section)}\n")
    assert str(ds) == "\nSummary.\n"


@pytest.mark.parametrize(
    "index, parsed, expected",
    [
        (
            ".. index:: default\n   :refguide: something, else, and more\n   text",
            {"default": "default", "refguide": ["something", "else", "and more"]},
            "\n.. index:: default\n   :refguide: something, else, and more",
        ),
        (
            ".. index::\n   :refguide: a, b",
            {"default": "", "refguide": ["a", "b"]},
            "\n.. index:: \n   :refguide: a, b",
        ),
        (".. index:: default", {"default": "default"}, "\n.. index:: default"),
        (".. index::", {"default": ""}, ""),
    ],
    ids=["default", "no-default", "default-only", "empty"],
)
def test_index_section(index, parsed, expected):
    # The index is a section that starts with ".. index::". Only its default
    # and its ":key: values" lines are kept, and an empty index is dropped
    nds = NumpyDocString(f"\nSummary.\n\n{index}\n")
    assert nds["index"] == parsed
    assert str(nds) == "\nSummary.\n" + expected


see_also = """
Summary.

See Also
--------
foo, :meth:`bar`, baz.qux
spam : The spam function,
    continued.

eggs
ham : The ham function.
"""


@pytest.mark.parametrize(
    "func_role, links",
    [
        ("", ["foo, :meth:`bar`, baz.qux", "spam", "eggs", "ham"]),
        (
            "func",
            [
                ":func:`foo`, :meth:`bar`, :func:`baz.qux`",
                ":func:`spam`",
                ":func:`eggs`",
                ":func:`ham`",
            ],
        ),
    ],
    ids=["no-role", "func-role"],
)
def test_see_also(func_role, links):
    nds = NumpyDocString(see_also)
    assert nds["See Also"] == [
        ([("foo", None), ("bar", "meth"), ("baz.qux", None)], []),
        ([("spam", None)], ["The spam function,", "continued."]),
        ([("eggs", None)], []),
        ([("ham", None)], ["The ham function."]),
    ]
    # An entry without a description gets the empty description ".."; the
    # default role is only given to the names without one
    expected = f"""
Summary.

See Also
--------
{links[0]}
    ..
{links[1]}
    The spam function, continued.
{links[2]}
    ..
{links[3]}
    The ham function.
"""
    assert nds.__str__(func_role=func_role) == expected


@pytest.mark.parametrize("line", ["foo bar", "foo:bar"])
def test_see_also_parse_error(line):
    doc = f"Summary.\n\nSee Also\n--------\n{line}\n"
    msg = f"{line} is not a item name"
    with pytest.raises(ParseError, match=f"^{re.escape(msg)}") as exc_info:
        Docstring(doc)
    # The docstring that failed to parse is attached to the error
    assert exc_info.value.docstring == doc
    assert str(exc_info.value) == f"{msg} in {doc}"


@pytest.mark.parametrize("line, loc", [("foo, : desc", 4), ("foo, bar, : desc", 9)])
def test_see_also_trailing_comma(line, loc):
    msg = (
        "Unexpected comma or period after function list at index "
        f'{loc} of line "{line}"'
    )
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        Docstring(f"Summary.\n\nSee Also\n--------\n{line}\n")


def test_empty_ds():
    ds = Docstring(None)
    ds.replace_block("summary", ["The is the new summary."])

    ds.remove_parameters("x")

    new = Parameter("w", "ndarray", ["An array input."])
    ds.insert_parameters("y", new)
    assert ds.extract_parameters("x") is None
    assert str(ds) == "None"


@pytest.mark.parametrize("docstring", ["", "   ", "\n"])
def test_empty_string_ds(docstring):
    # GH#9765: Nuitka sets __doc__ to "" instead of None.
    # Empty or whitespace-only docstrings should behave like None.
    ds = Docstring(docstring)
    assert ds._docstring is None
    ds.replace_block("summary", ["New summary."])
    ds.remove_parameters("x")
    new = Parameter("w", "ndarray", ["An array input."])
    ds.insert_parameters("y", new)
    assert ds.extract_parameters("x") is None
    assert str(ds) == "None"


@pytest.mark.parametrize(
    "docstring, msg",
    [
        (bad_yields, "Docstring contains both a Returns and Yields section."),
        (bad_receives, "Docstring contains a Receives section but not Yields."),
    ],
    ids=["returns-yields", "receives"],
)
def test_yield_return(docstring, msg):
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        Docstring(docstring)


def test_yields_receives():
    doc = """
Summary.

Receives
--------
x : int
    The value sent.

Yields
------
int
    The value yielded.
"""
    # Yields is written before Receives
    expected = """
Summary.

Yields
------
int
    The value yielded.

Receives
--------
x : int
    The value sent.
"""
    assert str(Docstring(doc)) == expected


def test_multiple_sig():
    # Of several signatures, each in its own paragraph, the last one is kept
    # and the rest of the docstring is parsed as without them
    nds = NumpyDocString(with_sig)
    assert nds["Signature"] == "func(x, y, z=1)"
    assert str(nds) == "func(x, y, z=1)\n" + str(NumpyDocString(good))


@pytest.mark.parametrize(
    "docstring, signature, expected",
    [
        ("func(x, *args)", "func(x, *args)", "func(x, \\*args)\n"),
        (
            "func(x)\n\nfunc(x, y)\n\nThe summary.\n",
            "func(x, y)",
            "func(x, y)\n\nThe summary.\n",
        ),
        ("A summary without a signature", "", "\nA summary without a signature\n"),
    ],
    ids=["signature-only", "last-of-several", "summary-only"],
)
def test_signature(docstring, signature, expected):
    # The last of several signatures is kept, and a "*" in it is escaped
    nds = NumpyDocString(docstring)
    assert nds["Signature"] == signature
    assert str(nds) == expected


def test_signature_before_section():
    nds = NumpyDocString("func(x)\n\nParameters\n----------\nx : int\n    The x.\n")
    assert nds["Signature"] == "func(x)"
    assert nds["Parameters"] == [Parameter("x", "int", ["The x."])]
