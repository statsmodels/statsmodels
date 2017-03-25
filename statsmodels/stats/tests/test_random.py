# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:40:25 2017

Author: Josef Perktold

"""
from __future__ import division

import numpy as np
from numpy.testing import assert_allclose
from statsmodels.stats._random import (p_table, simulate_table_permutation_gen,
                                       simulate_table_conditional)

def check_table_distribution(res_mc):
    k_rows, k_cols = res_mc[0].shape
    res_mc2 = res_mc.reshape(res_mc.shape[0], -1)
    uni = set(tuple(i) for i in res_mc2)
    uniques = np.asarray(list(uni))
    for u in uniques:
        prob = p_table(u.reshape(k_rows, k_cols))
        freq = (res_mc2 == u).all(1).mean()

    return prob, freq


class CheckRandomTable(object):

    def test_random(self):
        n_repl = 1000
        np.random.seed(987126)
        res_mc = [simulate_table_conditional(self.n_row, self.n_col) for i in range(n_repl)]
        res_mc = np.asarray(res_mc)
        prob, freq = check_table_distribution(res_mc)
        # maybe need tol based on standard deviation of proportion
        assert_allclose(freq, prob, rtol=0.5)


    def test_random_gen(self):
        n_repl = 1000
        np.random.seed(987126)
        it = simulate_table_permutation_gen(self.n_row, self.n_col)
        res_mc = [next(it) for _ in range(n_repl)]
        res_mc = np.asarray(res_mc)
        prob, freq = check_table_distribution(res_mc)
        # maybe need tol based on standard deviation of proportion
        assert_allclose(freq, prob, rtol=0.5)


class TestRandomTable1(CheckRandomTable):

    @classmethod
    def setup_class(cls):
        cls.n_row = [2, 4, 1]
        cls.n_col = [4, 1, 2]


class TestRandomTable2(CheckRandomTable):

    @classmethod
    def setup_class(cls):
        cls.n_row = [2, 4, 1]
        cls.n_col = [4, 1, 2]
