import numpy as np
from numpy.testing import assert_equal
import pandas as pd
import pytest

from statsmodels.stats.base import AllPairsResults, HolderTuple


def test_holdertuple():
    with pytest.warns(FutureWarning, match="HolderTuple is deprecated"):
        ht = HolderTuple(statistic=5, pvalue=0.1, text="just something",
                         extra=[1, 2, 4])
    assert_equal(len(ht), 2)
    assert_equal(ht[:], [5, 0.1])
    p, v = ht
    assert_equal([p, v], [5, 0.1])
    p, v = ht[0], ht[1]
    assert_equal([p, v], [5, 0.1])
    assert_equal(list(ht), [5, 0.1])
    assert_equal(np.asarray(ht), [5, 0.1])
    assert_equal(np.asarray(ht).dtype, np.float64)
    x = np.zeros((2, 2))
    x[0] = ht
    assert_equal(x, [[5, 0.1], [0, 0]])

    assert_equal(pd.Series(ht).values, [5, 0.1])
    assert_equal(pd.DataFrame([ht, ht]).values, [[5, 0.1], [5, 0.1]])

    assert_equal(ht.statistic, 5)
    assert_equal(ht.pvalue, 0.1)
    assert_equal(ht.extra, [1, 2, 4])
    assert_equal(ht.text, "just something")


def test_allpairsresults_summary_pval_table():
    pvals_raw = np.array([0.01, 0.2, 0.03])
    all_pairs = [(0, 1), (0, 2), (1, 2)]
    levels = ["a", "b", "c"]
    res = AllPairsResults(pvals_raw, all_pairs, levels=levels)

    import statsmodels.stats.multitest as smt
    expected_corrected = smt.multipletests(pvals_raw, method="hs")[1]
    assert_equal(res.pval_corrected(), expected_corrected)

    table = res.pval_table()
    assert table.shape == (3, 3)
    for (i, j), p in zip(all_pairs, expected_corrected, strict=True):
        assert_equal(table[i, j], p)
    # entries with no corresponding pair stay at the default fill value
    assert table[1, 0] == 0

    text = res.summary()
    assert str(res) == text
    for name, p in zip(res.all_pairs_names, expected_corrected, strict=True):
        assert name in text
        assert f"{p:6.4g}" in text
    assert res.all_pairs_names == ["a-b", "a-c", "b-c"]


def test_holdertuple2():
    with pytest.warns(FutureWarning, match="HolderTuple is deprecated"):
        ht = HolderTuple(tuple_=("statistic", "extra"), statistic=5, pvalue=0.1,
                         text="just something", extra=[1, 2, 4])
    assert_equal(len(ht), 2)
    assert_equal(ht[:], [5, [1, 2, 4]])
    p, v = ht
    assert_equal([p, v], [5, [1, 2, 4]])
    p, v = ht[0], ht[1]
    assert_equal([p, v], [5, [1, 2, 4]])
    assert_equal(list(ht), [5, [1, 2, 4]])

    x = np.asarray(ht, dtype=object)
    assert_equal(x, np.asarray([5, [1, 2, 4]], dtype=object))
    assert_equal(x.dtype, np.dtype("O"))
    # assert_equal(pd.Series(ht).values, [5, [1, 2, 4]])
    # assert_equal(pd.Series(ht).dtype, np.dtype('O'))

    assert_equal(ht.statistic, 5)
    assert_equal(ht.pvalue, 0.1)
    assert_equal(ht.extra, [1, 2, 4])
    assert_equal(ht.text, "just something")
