"""
Tests corresponding to sandbox.tools.cross_val
"""
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
import pytest

from statsmodels.sandbox.tools.cross_val import (
    KFold,
    KStepAhead,
    LeaveOneLabelOut,
    LeaveOneOut,
    LeavePOut,
    split,
)


def test_leave_one_out():
    loo = LeaveOneOut(4)
    folds = list(loo)
    assert_equal(len(folds), 4)
    for i, (train, test) in enumerate(folds):
        assert_equal(test.sum(), 1)
        assert test[i]
        # train/test partition the full index set with no overlap
        assert_array_equal(train, ~test)
        assert_equal(train.sum() + test.sum(), 4)


def test_leave_one_out_repr():
    loo = LeaveOneOut(4)
    assert repr(loo) == "statsmodels.sandbox.tools.cross_val.LeaveOneOut(n=4)"


def test_leave_p_out():
    lpo = LeavePOut(4, 2)
    folds = list(lpo)
    # C(4, 2) == 6 combinations
    assert_equal(len(folds), 6)
    for train, test in folds:
        assert_equal(test.sum(), 2)
        assert_equal(train.sum(), 2)
        assert_array_equal(train, ~test)
    # every 2-combination of indices appears as a test set exactly once
    seen = {tuple(np.flatnonzero(test)) for _, test in folds}
    assert_equal(len(seen), 6)


def test_leave_p_out_repr():
    lpo = LeavePOut(4, 2)
    assert repr(lpo) == "statsmodels.sandbox.tools.cross_val.LeavePOut(n=4, p=2)"


def test_kfold_partitions_all_indices_without_overlap():
    # regression test for the fold-size formula, not just the total count
    for n, k in [(10, 3), (5, 2), (7, 4), (12, 5)]:
        kf = KFold(n, k)
        folds = list(kf)
        assert_equal(len(folds), k)
        seen = np.zeros(n, dtype=int)
        for train, test in folds:
            assert_array_equal(train, ~test)
            seen += test.astype(int)
        # each index appears in exactly one test fold
        assert_array_equal(seen, np.ones(n, dtype=int))


def test_kfold_fold_sizes_use_ceil_not_trunc():
    # KFold's docstring claims "All the folds have size trunc(n/k), the
    # last one has the complementary", but the implementation computes
    # j = ceil(n / k) and gives the *last* fold the (smaller) remainder,
    # not trunc(n/k). Document the actual behavior here since it disagrees
    # with the docstring.
    kf = KFold(10, 3)
    sizes = [int(test.sum()) for _, test in kf]
    assert_equal(sizes, [4, 4, 2])  # ceil(10/3) == 4, not trunc(10/3) == 3

    kf = KFold(5, 2)
    sizes = [int(test.sum()) for _, test in kf]
    assert_equal(sizes, [3, 2])  # ceil(5/2) == 3, not trunc(5/2) == 2


def test_kfold_k_equal_one_is_degenerate():
    # k=1 is accepted by the k > 0 check, but produces a single fold where
    # the entire sample is the test set and nothing is left to train on.
    kf = KFold(5, 1)
    folds = list(kf)
    assert_equal(len(folds), 1)
    train, test = folds[0]
    assert_equal(train.sum(), 0)
    assert_equal(test.sum(), 5)


def test_kfold_invalid_k_raises_assertion_error_not_value_error():
    # BUG: KFold validates its arguments with
    #     assert k > 0, ValueError("cannot have k below 1")
    # This constructs a ValueError but never raises it; `assert` raises
    # AssertionError using that ValueError instance as its message. So
    # callers doing `except ValueError` will NOT catch invalid k, and
    # running Python with -O strips the assertions entirely, silently
    # disabling this validation. This test documents the current
    # (buggy) behavior; it should be updated if the bug is fixed to
    # actually `raise ValueError(...)` instead.
    with pytest.raises(AssertionError):
        KFold(5, 0)
    with pytest.raises(AssertionError):
        KFold(5, 5)
    with pytest.raises(AssertionError):
        KFold(5, -1)

    with pytest.raises(AssertionError) as exc_info:
        KFold(5, 0)
    assert isinstance(exc_info.value.args[0], ValueError)


def test_kfold_repr():
    kf = KFold(5, 2)
    assert repr(kf) == "statsmodels.sandbox.tools.cross_val.KFold(n=5, k=2)"


def test_leave_one_label_out():
    labels = [1, 1, 2, 2, 3]
    lol = LeaveOneLabelOut(labels)
    folds = list(lol)
    assert_equal(len(folds), 3)  # 3 unique labels
    for train, test in folds:
        assert_array_equal(train, ~test)
    # the label-2 fold selects exactly indices 2 and 3
    _, test_for_label_2 = folds[1]
    assert_array_equal(np.flatnonzero(test_for_label_2), [2, 3])


def test_leave_one_label_out_does_not_mutate_input():
    labels = [1, 1, 2, 2, 3]
    lol = LeaveOneLabelOut(labels)
    list(lol)
    assert_equal(labels, [1, 1, 2, 2, 3])


def test_leave_one_label_out_repr():
    labels = [1, 1, 2]
    lol = LeaveOneLabelOut(labels)
    assert repr(lol) == (
        "statsmodels.sandbox.tools.cross_val.LeaveOneLabelOut(labels=[1, 1, 2])"
    )


def test_split():
    x = np.arange(10).reshape(5, 2)
    y = np.arange(5)
    train = np.array([True, True, False, False, True])
    test = ~train

    x_train, x_test, y_train, y_test = split(train, test, x, y)
    assert_array_equal(x_train, x[train])
    assert_array_equal(x_test, x[test])
    assert_array_equal(y_train, y[train])
    assert_array_equal(y_test, y[test])


def test_split_single_array():
    x = np.arange(5)
    train = np.array([True, False, True, False, True])
    test = ~train
    x_train, x_test = split(train, test, x)
    assert_array_equal(x_train, [0, 2, 4])
    assert_array_equal(x_test, [1, 3])


def test_split_accepts_list_input():
    # split() calls np.asanyarray on each arg, so plain lists should work
    x_train, x_test = split(
        np.array([True, False, True]), np.array([False, True, False]), [10, 20, 30]
    )
    assert_array_equal(x_train, [10, 30])
    assert_array_equal(x_test, [20])


def test_kstepahead_slice_mode_default_kall():
    ks = KStepAhead(10, k=2, start=5)
    folds = list(ks)
    assert_equal(len(folds), 3)  # range(start, n - k) == range(5, 8)
    train_slice, test_slice = folds[0]
    assert_equal(train_slice, slice(None, 5, None))
    assert_equal(test_slice, slice(5, 7, None))  # kall=True: both steps
    train_slice, test_slice = folds[-1]
    assert_equal(train_slice, slice(None, 7, None))
    assert_equal(test_slice, slice(7, 9, None))


def test_kstepahead_slice_mode_kall_false():
    # kall=False: only the k-th step ahead is in the test slice
    ks = KStepAhead(10, k=2, start=5, kall=False)
    train_slice, test_slice = next(iter(ks))
    assert_equal(train_slice, slice(None, 5, None))
    assert_equal(test_slice, slice(6, 7, None))


def test_kstepahead_boolean_mode_matches_slice_mode():
    ks_slices = KStepAhead(10, k=2, start=5, return_slice=True)
    ks_bools = KStepAhead(10, k=2, start=5, return_slice=False)
    n = 10
    for (train_slice, test_slice), (train_bool, test_bool) in zip(
        ks_slices, ks_bools, strict=True
    ):
        expected_train = np.zeros(n, dtype=bool)
        expected_train[train_slice] = True
        expected_test = np.zeros(n, dtype=bool)
        expected_test[test_slice] = True
        assert_array_equal(train_bool, expected_train)
        assert_array_equal(test_bool, expected_test)


def test_kstepahead_default_start_is_quarter_of_n():
    # start=None defaults to trunc(n * 0.25)
    ks = KStepAhead(20, k=1)
    assert_equal(ks.start, 5)
    first_train_slice, _ = next(iter(ks))
    assert_equal(first_train_slice, slice(None, 5, None))


def test_kstepahead_repr():
    ks = KStepAhead(10, k=2, start=5)
    assert repr(ks) == "statsmodels.sandbox.tools.cross_val.KStepAhead(n=10)"
