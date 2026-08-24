import numpy as np
import pandas as pd
import pytest

from statsmodels.tools.testing import (
    Holder,
    MarginTableTestBunch,
    ParamsTableTestBunch,
    assert_equal,
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
