import pytest

from statsmodels.graphics.utils import _import_mpl

try:
    import matplotlib as mpl  # noqa: F401

    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False


def _call_import_mpl():
    # A plain module-level function that calls _import_mpl directly, so
    # the expected caller name in the error message below is deterministic.
    return _import_mpl()


class _ImportMplCaller:
    def call_import_mpl(self):
        return _import_mpl()


def _call_import_mpl_stacklevel_2():
    # Simulates a wrapper (like create_mpl_ax) that calls _import_mpl on
    # behalf of its own caller, one frame further up.
    return _via_wrapper()


def _via_wrapper():
    # Mirrors how create_mpl_ax/create_mpl_fig call _import_mpl on behalf
    # of their own caller, via stacklevel=2.
    return _import_mpl(stacklevel=2)


@pytest.mark.xfail(
    HAVE_MATPLOTLIB,
    reason="matplotlib is installed, so _import_mpl succeeds and never raises",
    strict=True,
)
def test_import_mpl_error_names_calling_function():
    with pytest.raises(ImportError) as exc_info:
        _call_import_mpl()
    expected = (
        f"{__name__}._call_import_mpl requires matplotlib "
        "but matplotlib is not installed."
    )
    assert str(exc_info.value) == expected


@pytest.mark.xfail(
    HAVE_MATPLOTLIB,
    reason="matplotlib is installed, so _import_mpl succeeds and never raises",
    strict=True,
)
def test_import_mpl_error_names_calling_method():
    caller = _ImportMplCaller()
    with pytest.raises(ImportError) as exc_info:
        caller.call_import_mpl()
    expected = (
        f"{__name__}._ImportMplCaller.call_import_mpl requires matplotlib "
        "but matplotlib is not installed."
    )
    assert str(exc_info.value) == expected


@pytest.mark.xfail(
    HAVE_MATPLOTLIB,
    reason="matplotlib is installed, so _import_mpl succeeds and never raises",
    strict=True,
)
def test_import_mpl_stacklevel_names_grandcaller():
    with pytest.raises(ImportError) as exc_info:
        _call_import_mpl_stacklevel_2()
    expected = (
        f"{__name__}._call_import_mpl_stacklevel_2 requires matplotlib "
        "but matplotlib is not installed."
    )
    assert str(exc_info.value) == expected
