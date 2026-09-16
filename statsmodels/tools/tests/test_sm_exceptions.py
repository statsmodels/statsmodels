import pytest

from statsmodels.tools.sm_exceptions import ParseError


def test_parse_error_str_without_docstring_attr():
    # documented contract: plain Exception-style message when no
    # `docstring` attribute has been attached
    err = ParseError("bad section")
    assert str(err) == "bad section"


def test_parse_error_str_with_docstring_attr():
    # documented contract: message is extended with "in <docstring>" once
    # a `docstring` attribute is attached (as statsmodels.tools.docstring
    # does when it catches and re-raises a ParseError)
    err = ParseError("bad section")
    err.docstring = "the offending docstring text"
    assert str(err) == "bad section in the offending docstring text"


def test_parse_error_is_exception_and_raisable():
    with pytest.raises(ParseError, match="boom") as excinfo:
        raise ParseError("boom")
    assert str(excinfo.value) == "boom"
