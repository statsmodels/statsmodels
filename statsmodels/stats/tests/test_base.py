
import numpy as np
from numpy.testing import assert_equal
import pandas as pd
from statsmodels.stats.base import HolderTuple


def test_holdertuple():
    ht = HolderTuple(statistic=5, pvalue=0.1, text="just something",
                     extra=[1, 2, 4])
    assert_equal(ht[:], [5, 0.1])
    p, v = ht
    assert_equal([p, v], [5, 0.1])
    p, v = ht[0], ht[1]
    assert_equal([p, v], [5, 0.1])
    assert_equal(list(ht), [5, 0.1])
    assert_equal(np.asarray(ht), [5, 0.1])
    assert_equal(np.asarray(ht).dtype, np.float64)
    assert_equal(pd.Series(ht).values, [5, 0.1])
    assert_equal(pd.DataFrame([ht, ht]).values, [[5, 0.1],[5, 0.1]])

    assert_equal(ht.statistic, 5)
    assert_equal(ht.pvalue, 0.1)
    assert_equal(ht.extra, [1, 2, 4])
    assert_equal(ht.text, "just something") 


def test_holdertuple2():
    ht = HolderTuple(tuple_=("statistic", "extra"), statistic=5, pvalue=0.1,
                     text="just something", extra=[1, 2, 4])
    assert_equal(ht[:], [5, [1, 2, 4]])
    p, v = ht
    assert_equal([p, v], [5, [1, 2, 4]])
    p, v = ht[0], ht[1]
    assert_equal([p, v], [5, [1, 2, 4]])
    assert_equal(list(ht), [5, [1, 2, 4]])
    assert_equal(np.asarray(ht), [5, [1, 2, 4]])
    assert_equal(np.asarray(ht).dtype, np.dtype('O'))
    assert_equal(pd.Series(ht).values, [5, [1, 2, 4]])
    assert_equal(pd.Series(ht).dtype, np.dtype('O'))

    assert_equal(ht.statistic, 5)
    assert_equal(ht.pvalue, 0.1)
    assert_equal(ht.extra, [1, 2, 4])
    assert_equal(ht.text, "just something")