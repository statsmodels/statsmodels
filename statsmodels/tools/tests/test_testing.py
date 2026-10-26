import warnings

import numpy as np
import pandas as pd
import pytest

from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tools.testing import (
    Holder,
    MarginTableTestBunch,
    ParamsTableTestBunch,
    assert_equal,
    warns_or_none,
)


@pytest.mark.parametrize(
    "attribute, bunch_type",
    [("params_table", ParamsTableTestBunch), ("margins_table", MarginTableTestBunch)],
)
def check_params_table_classes(attribute, bunch_type):
    table = np.empty((10, 4))
    bunch = bunch_type(**{attribute: table})
    assert attribute in bunch


def test_bad_table():
    table = np.empty((10, 4))
    with pytest.raises(AttributeError):
        ParamsTableTestBunch(margins_table=table)


def test_holder():
    holder = Holder()
    holder.new_attr = 1
    assert hasattr(holder, "new_attr")
    assert holder.new_attr == 1


def test_assert_equal_forwards_err_msg():
    # GH: the non-pandas branch hardcoded err_msg="" and verbose=True
    # instead of forwarding the caller's arguments.
    with pytest.raises(AssertionError, match="my custom message"):
        assert_equal(1, 2, err_msg="my custom message")


def test_assert_equal_index_forwards_kwds():
    # GH: the pd.Index branch dropped **kwds entirely, even though the
    # docstring documents it as being passed through for Index too.
    left = pd.Index([1, 2, 3], name="a")
    right = pd.Index([1, 2, 3], name="b")
    with pytest.raises(AssertionError):
        assert_equal(left, right)
    assert_equal(left, right, check_names=False)


def test_warns_or_none():
    # no warning and the allowed warning pass
    with warns_or_none(RuntimeWarning):
        pass
    with warns_or_none(RuntimeWarning, match="platform"):
        warnings.warn("platform dependent", RuntimeWarning)
    # a subclass and one of several categories are allowed
    with warns_or_none((FutureWarning, UserWarning)):
        warnings.warn("subclass", ConvergenceWarning)
        warnings.warn("future", FutureWarning)


def test_warns_or_none_other_warning():
    # warnings that are not allowed are emitted when the block ends
    with pytest.warns(UserWarning, match="other category"):
        with warns_or_none(RuntimeWarning):
            warnings.warn("other category", UserWarning)
    with pytest.warns(RuntimeWarning, match="other message"):
        with warns_or_none(RuntimeWarning, match="platform"):
            warnings.warn("platform dependent", RuntimeWarning)
            warnings.warn("other message", RuntimeWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="not allowed"):
            with warns_or_none(RuntimeWarning):
                warnings.warn("not allowed", UserWarning)
