"""
Test functions for models.tools
"""
from statsmodels.compat.pandas import assert_frame_equal, assert_series_equal
from statsmodels.compat.python import lrange

import string

import numpy as np
from numpy.random import standard_normal
from numpy.testing import (assert_equal, assert_array_equal,
                           assert_almost_equal, assert_string_equal)
import pandas as pd
import pytest

from statsmodels.datasets import longley
from statsmodels.tools import tools
from statsmodels.tools.tools import pinv_extended


@pytest.fixture(scope='module')
def string_var():
    string_var = [string.ascii_lowercase[0:5],
                  string.ascii_lowercase[5:10],
                  string.ascii_lowercase[10:15],
                  string.ascii_lowercase[15:20],
                  string.ascii_lowercase[20:25]]
    string_var *= 5
    string_var = np.asarray(sorted(string_var))
    series = pd.Series(string_var, name='string_var')
    return series


class TestTools(object):

    def test_add_constant_list(self):
        x = lrange(1,5)
        x = tools.add_constant(x)
        y = np.asarray([[1,1,1,1],[1,2,3,4.]]).T
        assert_equal(x, y)

    def test_add_constant_1d(self):
        x = np.arange(1,5)
        x = tools.add_constant(x)
        y = np.asarray([[1,1,1,1],[1,2,3,4.]]).T
        assert_equal(x, y)

    def test_add_constant_has_constant1d(self):
        x = np.ones(5)
        x = tools.add_constant(x, has_constant='skip')
        assert_equal(x, np.ones((5,1)))

        with pytest.raises(ValueError):
            tools.add_constant(x, has_constant='raise')

        assert_equal(tools.add_constant(x, has_constant='add'),
                     np.ones((5, 2)))

    def test_add_constant_has_constant2d(self):
        x = np.asarray([[1,1,1,1],[1,2,3,4.]]).T
        y = tools.add_constant(x, has_constant='skip')
        assert_equal(x, y)

        with pytest.raises(ValueError):
            tools.add_constant(x, has_constant='raise')

        assert_equal(tools.add_constant(x, has_constant='add'),
                     np.column_stack((np.ones(4), x)))

    def test_add_constant_recarray(self):
        dt = np.dtype([('', int), ('', '<S4'), ('', np.float32), ('', np.float64)])
        x = np.array([(1, 'abcd', 1.0, 2.0),
                      (7, 'abcd', 2.0, 4.0),
                      (21, 'abcd', 2.0, 8.0)], dt)
        x = x.view(np.recarray)
        with pytest.warns(FutureWarning, match="recarray support"):
            y = tools.add_constant(x)
        assert_equal(y['const'],np.array([1.0,1.0,1.0]))
        for f in x.dtype.fields:
            assert y[f].dtype == x[f].dtype

    def test_add_constant_series(self):
        s = pd.Series([1.0,2.0,3.0])
        output = tools.add_constant(s)
        expected = pd.Series([1.0,1.0,1.0],name='const')
        assert_series_equal(expected, output['const'])

    def test_add_constant_dataframe(self):
        df = pd.DataFrame([[1.0, 'a', 4], [2.0, 'bc', 9], [3.0, 'def', 16]])
        output = tools.add_constant(df)
        expected = pd.Series([1.0, 1.0, 1.0], name='const')
        assert_series_equal(expected, output['const'])
        dfc = df.copy()
        dfc.insert(0, 'const', np.ones(3))
        assert_frame_equal(dfc, output)

    def test_add_constant_zeros(self):
        a = np.zeros(100)
        output = tools.add_constant(a)
        assert_equal(output[:,0],np.ones(100))

        s = pd.Series([0.0,0.0,0.0])
        output = tools.add_constant(s)
        expected = pd.Series([1.0, 1.0, 1.0], name='const')
        assert_series_equal(expected, output['const'])

        df = pd.DataFrame([[0.0, 'a', 4], [0.0, 'bc', 9], [0.0, 'def', 16]])
        output = tools.add_constant(df)
        dfc = df.copy()
        dfc.insert(0, 'const', np.ones(3))
        assert_frame_equal(dfc, output)

        df = pd.DataFrame([[1.0, 'a', 0], [0.0, 'bc', 0], [0.0, 'def', 0]])
        output = tools.add_constant(df)
        dfc = df.copy()
        dfc.insert(0, 'const', np.ones(3))
        assert_frame_equal(dfc, output)

    def test_recipr(self):
        X = np.array([[2,1],[-1,0]])
        Y = tools.recipr(X)
        assert_almost_equal(Y, np.array([[0.5,1],[0,0]]))

    def test_recipr0(self):
        X = np.array([[2,1],[-4,0]])
        Y = tools.recipr0(X)
        assert_almost_equal(Y, np.array([[0.5,1],[-0.25,0]]))

    def test_extendedpinv(self):
        X = standard_normal((40, 10))
        np_inv = np.linalg.pinv(X)
        np_sing_vals = np.linalg.svd(X, 0, 0)
        sm_inv, sing_vals = pinv_extended(X)
        assert_almost_equal(np_inv, sm_inv)
        assert_almost_equal(np_sing_vals, sing_vals)

    def test_extendedpinv_singular(self):
        X = standard_normal((40, 10))
        X[:, 5] = X[:, 1] + X[:, 3]
        np_inv = np.linalg.pinv(X)
        np_sing_vals = np.linalg.svd(X, 0, 0)
        sm_inv, sing_vals = pinv_extended(X)
        assert_almost_equal(np_inv, sm_inv)
        assert_almost_equal(np_sing_vals, sing_vals)

    def test_fullrank(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = standard_normal((40,10))
            X[:,0] = X[:,1] + X[:,2]

            Y = tools.fullrank(X)
            assert_equal(Y.shape, (40,9))

            X[:,5] = X[:,3] + X[:,4]
            Y = tools.fullrank(X)
            assert_equal(Y.shape, (40,8))
            warnings.simplefilter("ignore")


def test_estimable():
    rng = np.random.RandomState(20120713)
    N, P = (40, 10)
    X = rng.normal(size=(N, P))
    C = rng.normal(size=(1, P))
    isestimable = tools.isestimable
    assert isestimable(C, X)
    assert isestimable(np.eye(P), X)
    for row in np.eye(P):
        assert isestimable(row, X)
    X = np.ones((40, 2))
    assert isestimable([1, 1], X)
    assert not isestimable([1, 0], X)
    assert not isestimable([0, 1], X)
    assert not isestimable(np.eye(2), X)
    halfX = rng.normal(size=(N, 5))
    X = np.hstack([halfX, halfX])
    assert not isestimable(np.hstack([np.eye(5), np.zeros((5, 5))]), X)
    assert not isestimable(np.hstack([np.zeros((5, 5)), np.eye(5)]), X)
    assert isestimable(np.hstack([np.eye(5), np.eye(5)]), X)
    # Test array_like for design
    XL = X.tolist()
    assert isestimable(np.hstack([np.eye(5), np.eye(5)]), XL)
    # Test ValueError for incorrect number of columns
    X = rng.normal(size=(N, 5))
    for n in range(1, 4):
        with pytest.raises(ValueError):
            isestimable(np.ones((n,)), X)
    with pytest.raises(ValueError):
        isestimable(np.eye(4), X)


class TestCategoricalNumerical(object):
    #TODO: use assert_raises to check that bad inputs are taken care of
    @classmethod
    def setup_class(cls):
        #import string
        stringabc = 'abcdefghijklmnopqrstuvwxy'
        cls.des = np.random.randn(25,2)
        cls.instr = np.floor(np.arange(10,60, step=2)/10)
        x=np.zeros((25,5))
        x[:5,0]=1
        x[5:10,1]=1
        x[10:15,2]=1
        x[15:20,3]=1
        x[20:25,4]=1
        cls.dummy = x
        structdes = np.zeros((25,1),dtype=[('var1', 'f4'),('var2', 'f4'),
                    ('instrument','f4'),('str_instr','a10')])
        structdes['var1'] = cls.des[:,0][:,None]
        structdes['var2'] = cls.des[:,1][:,None]
        structdes['instrument'] = cls.instr[:,None]
        string_var = [stringabc[0:5], stringabc[5:10],
                stringabc[10:15], stringabc[15:20],
                stringabc[20:25]]
        string_var *= 5
        cls.string_var = np.array(sorted(string_var))
        structdes['str_instr'] = cls.string_var[:,None]
        cls.structdes = structdes
        cls.recdes = structdes.view(np.recarray)

    def test_array2d(self):
        des = np.column_stack((self.des, self.instr, self.des))
        des = tools.categorical(des, col=2)
        assert_array_equal(des[:,-5:], self.dummy)
        assert_equal(des.shape[1],10)

    def test_array1d(self):
        des = tools.categorical(self.instr)
        assert_array_equal(des[:, -5:], self.dummy)
        assert_equal(des.shape[1], 6)

    def test_array1d_col_error(self):
        with pytest.raises(TypeError, match='col must be a str, int or None'):
            tools.categorical(self.instr, col={'a': 1})

    def test_array2d_drop(self):
        des = np.column_stack((self.des, self.instr, self.des))
        des = tools.categorical(des, col=2, drop=True)
        assert_array_equal(des[:,-5:], self.dummy)
        assert_equal(des.shape[1],9)

    def test_array1d_drop(self):
        des = tools.categorical(self.instr, drop=True)
        assert_array_equal(des, self.dummy)
        assert_equal(des.shape[1],5)

    def test_recarray2d(self):
        with pytest.warns(FutureWarning, match="recarray support"):
            des = tools.categorical(self.recdes, col='instrument')
        # better way to do this?
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 9)

    def test_recarray2d_error(self):
        arr = np.c_[self.recdes, self.recdes]
        with pytest.raises(IndexError, match='col is None and the input'):
            with pytest.warns(FutureWarning, match="recarray support"):
                tools.categorical(arr, col=None)

    def test_recarray2dint(self):
        with pytest.warns(FutureWarning, match="recarray support"):
            des = tools.categorical(self.recdes, col=2)
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 9)

    def test_recarray1d(self):
        instr = self.structdes['instrument'].view(np.recarray)
        with pytest.warns(FutureWarning, match="recarray support"):
            dum = tools.categorical(instr)
        test_dum = np.column_stack(([dum[_] for _ in dum.dtype.names[-5:]]))
        assert_array_equal(test_dum, self.dummy)
        assert_equal(len(dum.dtype.names), 6)

    def test_recarray1d_drop(self):
        instr = self.structdes['instrument'].view(np.recarray)
        with pytest.warns(FutureWarning, match="recarray support"):
            dum = tools.categorical(instr, drop=True)
        test_dum = np.column_stack(([dum[_] for _ in dum.dtype.names]))
        assert_array_equal(test_dum, self.dummy)
        assert_equal(len(dum.dtype.names), 5)

    def test_recarray2d_drop(self):
        with pytest.warns(FutureWarning, match="recarray support"):
            des = tools.categorical(self.recdes, col='instrument', drop=True)
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 8)

    def test_structarray2d(self):
        with pytest.warns(FutureWarning, match="recarray support"):
            des = tools.categorical(self.structdes, col='instrument')
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 9)

    def test_structarray2dint(self):
        with pytest.warns(FutureWarning, match="recarray support"):
            des = tools.categorical(self.structdes, col=2)
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 9)

    def test_structarray1d(self):
        instr = self.structdes['instrument'].view(dtype=[('var1', 'f4')])
        with pytest.warns(FutureWarning, match="recarray support"):
            dum = tools.categorical(instr)
        test_dum = np.column_stack(([dum[_] for _ in dum.dtype.names[-5:]]))
        assert_array_equal(test_dum, self.dummy)
        assert_equal(len(dum.dtype.names), 6)

    def test_structarray2d_drop(self):
        with pytest.warns(FutureWarning, match="recarray support"):
            des = tools.categorical(self.structdes, col='instrument', drop=True)
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 8)

    def test_structarray1d_drop(self):
        instr = self.structdes['instrument'].view(dtype=[('var1', 'f4')])
        with pytest.warns(FutureWarning, match="recarray support"):
            dum = tools.categorical(instr, drop=True)
        test_dum = np.column_stack(([dum[_] for _ in dum.dtype.names]))
        assert_array_equal(test_dum, self.dummy)
        assert_equal(len(dum.dtype.names), 5)

    @pytest.mark.xfail(reason="Call to tools.categorical raises "
                              "AttributeError.  Previously this was "
                              "commented-out with the comment "
                              "'comment out until we have type coercion'.",
                       strict=True, raises=AttributeError)
    def test_arraylike2d(self):
        des = tools.categorical(self.structdes.tolist(), col=2)
        test_des = des[:,-5:]
        assert_array_equal(test_des, self.dummy)
        assert_equal(des.shape[1], 9)

    @pytest.mark.xfail(reason="Call to tools.categorical raises "
                              "AttributeError.  Previously this was "
                              "commented-out with the comment "
                              "'comment out until we have type coercion'.",
                       strict=True, raises=AttributeError)
    def test_arraylike1d(self):
        instr = self.structdes['instrument'].tolist()
        dum = tools.categorical(instr)
        test_dum = dum[:,-5:]
        assert_array_equal(test_dum, self.dummy)
        assert_equal(dum.shape[1], 6)

    @pytest.mark.xfail(reason="Call to tools.categorical raises "
                              "AttributeError.  Previously this was "
                              "commented-out with the comment "
                              "'comment out until we have type coercion'.",
                       strict=True, raises=AttributeError)
    def test_arraylike2d_drop(self):
        des = tools.categorical(self.structdes.tolist(), col=2, drop=True)
        test_des = des[:,-5:]
        assert_array_equal(test_des, self.dummy)
        assert_equal(des.shape[1], 8)

    @pytest.mark.xfail(reason="Call to tools.categorical raises "
                              "AttributeError.  Previously this was "
                              "commented-out with the comment "
                              "'comment out until we have type coercion'.",
                       strict=True, raises=AttributeError)
    def test_arraylike1d_drop(self):
        instr = self.structdes['instrument'].tolist()
        dum = tools.categorical(instr, drop=True)
        assert_array_equal(dum, self.dummy)
        assert_equal(dum.shape[1], 5)


class TestCategoricalString(TestCategoricalNumerical):

    @pytest.mark.xfail(reason="No idea!  But xfailing instead of leaving "
                              "this commented out with the comment "
                              "'comment out until we have type coercion'",
                              strict=False)
    def test_array2d(self):
        des = np.column_stack((self.des, self.instr, self.des))
        des = tools.categorical(des, col=2)
        assert_array_equal(des[:, -5:], self.dummy)
        assert_equal(des.shape[1], 10)

    @pytest.mark.xfail(reason="No idea!  But xfailing instead of leaving "
                              "this commented out with the comment "
                              "'comment out until we have type coercion'",
                              strict=False)
    def test_array1d(self):
        des = tools.categorical(self.instr)
        assert_array_equal(des[:, -5:], self.dummy)
        assert_equal(des.shape[1], 6)

    @pytest.mark.xfail(reason="No idea!  But xfailing instead of leaving "
                              "this commented out with the comment "
                              "'comment out until we have type coercion'",
                              strict=False)
    def test_array2d_drop(self):
        des = np.column_stack((self.des, self.instr, self.des))
        des = tools.categorical(des, col=2, drop=True)
        assert_array_equal(des[:, -5:], self.dummy)
        assert_equal(des.shape[1], 9)

    def test_array1d_drop(self):
        des = tools.categorical(self.string_var, drop=True)
        assert_array_equal(des, self.dummy)
        assert_equal(des.shape[1], 5)

    def test_recarray2d(self):
        with pytest.warns(FutureWarning, match="recarray support"):
            des = tools.categorical(self.recdes, col='str_instr')
        # TODO: better way to do this?
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 9)

    def test_recarray2dint(self):
        with pytest.warns(FutureWarning, match="recarray support"):
            des = tools.categorical(self.recdes, col=3)
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 9)

    def test_recarray1d(self):
        instr = self.structdes['str_instr'].view(np.recarray)
        with pytest.warns(FutureWarning, match="recarray support"):
            dum = tools.categorical(instr)
        test_dum = np.column_stack(([dum[_] for _ in dum.dtype.names[-5:]]))
        assert_array_equal(test_dum, self.dummy)
        assert_equal(len(dum.dtype.names), 6)

    def test_recarray1d_drop(self):
        instr = self.structdes['str_instr'].view(np.recarray)
        with pytest.warns(FutureWarning, match="recarray support"):
            dum = tools.categorical(instr, drop=True)
        test_dum = np.column_stack(([dum[_] for _ in dum.dtype.names]))
        assert_array_equal(test_dum, self.dummy)
        assert_equal(len(dum.dtype.names), 5)

    def test_recarray2d_drop(self):
        with pytest.warns(FutureWarning, match="recarray support"):
            des = tools.categorical(self.recdes, col='str_instr', drop=True)
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 8)

    def test_structarray2d(self):
        with pytest.warns(FutureWarning, match="recarray support"):
            des = tools.categorical(self.structdes, col='str_instr')
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 9)

    def test_structarray2dint(self):
        with pytest.warns(FutureWarning, match="recarray support"):
            des = tools.categorical(self.structdes, col=3)
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 9)

    def test_structarray1d(self):
        instr = self.structdes['str_instr'].view(dtype=[('var1', 'a10')])
        with pytest.warns(FutureWarning, match="recarray support"):
            dum = tools.categorical(instr)
        test_dum = np.column_stack(([dum[_] for _ in dum.dtype.names[-5:]]))
        assert_array_equal(test_dum, self.dummy)
        assert_equal(len(dum.dtype.names), 6)

    def test_structarray2d_drop(self):
        with pytest.warns(FutureWarning, match="recarray support"):
            des = tools.categorical(self.structdes, col='str_instr', drop=True)
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 8)

    def test_structarray1d_drop(self):
        instr = self.structdes['str_instr'].view(dtype=[('var1', 'a10')])
        with pytest.warns(FutureWarning, match="recarray support"):
            dum = tools.categorical(instr, drop=True)
        test_dum = np.column_stack(([dum[_] for _ in dum.dtype.names]))
        assert_array_equal(test_dum, self.dummy)
        assert_equal(len(dum.dtype.names), 5)

    @pytest.mark.xfail(reason="Test has not been implemented for this class.",
                       strict=True, raises=NotImplementedError)
    def test_arraylike2d(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test has not been implemented for this class.",
                       strict=True, raises=NotImplementedError)
    def test_arraylike1d(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test has not been implemented for this class.",
                       strict=True, raises=NotImplementedError)
    def test_arraylike2d_drop(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test has not been implemented for this class.",
                       strict=True, raises=NotImplementedError)
    def test_arraylike1d_drop(self):
        raise NotImplementedError


def test_rec_issue302():
    arr = np.rec.fromrecords([[10], [11]], names='group')
    with pytest.warns(FutureWarning, match="recarray support"):
        actual = tools.categorical(arr)
    expected = np.rec.array([(10, 1.0, 0.0), (11, 0.0, 1.0)],
        dtype=[('group', int), ('group_10', float), ('group_11', float)])
    assert_array_equal(actual, expected)

def test_issue302():
    arr = np.rec.fromrecords([[10, 12], [11, 13]], names=['group', 'whatever'])
    with pytest.warns(FutureWarning, match="recarray support"):
        actual = tools.categorical(arr, col=['group'])
    expected = np.rec.array([(10, 12, 1.0, 0.0), (11, 13, 0.0, 1.0)],
        dtype=[('group', int), ('whatever', int), ('group_10', float),
               ('group_11', float)])
    assert_array_equal(actual, expected)

def test_pandas_const_series():
    dta = longley.load_pandas()
    series = dta.exog['GNP']
    series = tools.add_constant(series, prepend=False)
    assert_string_equal('const', series.columns[1])
    assert_equal(series.var(0)[1], 0)

def test_pandas_const_series_prepend():
    dta = longley.load_pandas()
    series = dta.exog['GNP']
    series = tools.add_constant(series, prepend=True)
    assert_string_equal('const', series.columns[0])
    assert_equal(series.var(0)[0], 0)

def test_pandas_const_df():
    dta = longley.load_pandas().exog
    dta = tools.add_constant(dta, prepend=False)
    assert_string_equal('const', dta.columns[-1])
    assert_equal(dta.var(0)[-1], 0)

def test_pandas_const_df_prepend():
    dta = longley.load_pandas().exog
    # regression test for #1025
    dta['UNEMP'] /= dta['UNEMP'].std()
    dta = tools.add_constant(dta, prepend=True)
    assert_string_equal('const', dta.columns[0])
    assert_equal(dta.var(0)[0], 0)


class TestNanDot(object):
    @classmethod
    def setup_class(cls):
        nan = np.nan
        cls.mx_1 = np.array([[nan, 1.], [2., 3.]])
        cls.mx_2 = np.array([[nan, nan], [2., 3.]])
        cls.mx_3 = np.array([[0., 0.], [0., 0.]])
        cls.mx_4 = np.array([[1., 0.], [1., 0.]])
        cls.mx_5 = np.array([[0., 1.], [0., 1.]])
        cls.mx_6 = np.array([[1., 2.], [3., 4.]])

    def test_11(self):
        test_res = tools.nan_dot(self.mx_1, self.mx_1)
        expected_res = np.array([[ np.nan,  np.nan], [ np.nan,  11.]])
        assert_array_equal(test_res, expected_res)

    def test_12(self):
        nan = np.nan
        test_res = tools.nan_dot(self.mx_1, self.mx_2)
        expected_res = np.array([[ nan,  nan], [ nan,  nan]])
        assert_array_equal(test_res, expected_res)

    def test_13(self):
        nan = np.nan
        test_res = tools.nan_dot(self.mx_1, self.mx_3)
        expected_res = np.array([[ 0.,  0.], [ 0.,  0.]])
        assert_array_equal(test_res, expected_res)

    def test_14(self):
        nan = np.nan
        test_res = tools.nan_dot(self.mx_1, self.mx_4)
        expected_res = np.array([[ nan,   0.], [  5.,   0.]])
        assert_array_equal(test_res, expected_res)

    def test_41(self):
        nan = np.nan
        test_res = tools.nan_dot(self.mx_4, self.mx_1)
        expected_res = np.array([[ nan,   1.], [ nan,   1.]])
        assert_array_equal(test_res, expected_res)

    def test_23(self):
        nan = np.nan
        test_res = tools.nan_dot(self.mx_2, self.mx_3)
        expected_res = np.array([[ 0.,  0.], [ 0.,  0.]])
        assert_array_equal(test_res, expected_res)

    def test_32(self):
        nan = np.nan
        test_res = tools.nan_dot(self.mx_3, self.mx_2)
        expected_res = np.array([[ 0.,  0.], [ 0.,  0.]])
        assert_array_equal(test_res, expected_res)

    def test_24(self):
        nan = np.nan
        test_res = tools.nan_dot(self.mx_2, self.mx_4)
        expected_res = np.array([[ nan,   0.], [  5.,   0.]])
        assert_array_equal(test_res, expected_res)

    def test_25(self):
        nan = np.nan
        test_res = tools.nan_dot(self.mx_2, self.mx_5)
        expected_res = np.array([[  0.,  nan], [  0.,   5.]])
        assert_array_equal(test_res, expected_res)

    def test_66(self):
        nan = np.nan
        test_res = tools.nan_dot(self.mx_6, self.mx_6)
        expected_res = np.array([[  7.,  10.], [ 15.,  22.]])
        assert_array_equal(test_res, expected_res)

class TestEnsure2d(object):
    @classmethod
    def setup_class(cls):
        x = np.arange(400.0).reshape((100,4))
        cls.df = pd.DataFrame(x, columns = ['a','b','c','d'])
        cls.series = cls.df.iloc[:,0]
        cls.ndarray = x

    def test_enfore_numpy(self):
        results = tools._ensure_2d(self.df, True)
        assert_array_equal(results[0], self.ndarray)
        assert_array_equal(results[1], self.df.columns)
        results = tools._ensure_2d(self.series, True)
        assert_array_equal(results[0], self.ndarray[:,[0]])
        assert_array_equal(results[1], self.df.columns[0])

    def test_pandas(self):
        results = tools._ensure_2d(self.df, False)
        assert_frame_equal(results[0], self.df)
        assert_array_equal(results[1], self.df.columns)

        results = tools._ensure_2d(self.series, False)
        assert_frame_equal(results[0], self.df.iloc[:,[0]])
        assert_equal(results[1], self.df.columns[0])

    def test_numpy(self):
        results = tools._ensure_2d(self.ndarray)
        assert_array_equal(results[0], self.ndarray)
        assert_equal(results[1], None)

        results = tools._ensure_2d(self.ndarray[:, 0])
        assert_array_equal(results[0], self.ndarray[:, [0]])
        assert_equal(results[1], None)


def test_categorical_pandas_errors(string_var):
    with pytest.raises(ValueError, match='data.name does not match col'):
        tools.categorical(string_var, 'unknown')

    df = pd.DataFrame(string_var)
    with pytest.raises(TypeError, match='col must be a str or int'):
        tools.categorical(df, None)
    with pytest.raises(ValueError, match='Column \'unknown\' not found in '
                                         'data'):
        tools.categorical(df, 'unknown')


def test_categorical_series(string_var):
    design = tools.categorical(string_var, drop=True)
    dummies = pd.get_dummies(pd.Categorical(string_var))
    assert_frame_equal(design, dummies)
    design = tools.categorical(string_var, drop=False)
    dummies.columns = list(dummies.columns)
    assert_frame_equal(design.iloc[:, :5], dummies)
    assert_series_equal(design.iloc[:, 5], string_var)
    _, dictnames = tools.categorical(string_var, drop=False, dictnames=True)
    for i, c in enumerate(pd.Categorical(string_var).categories):
        assert i in dictnames
        assert dictnames[i] == c


def test_categorical_dataframe(string_var):
    df = pd.DataFrame(string_var)
    design = tools.categorical(df, 'string_var', drop=True)
    dummies = pd.get_dummies(pd.Categorical(string_var))
    assert_frame_equal(design, dummies)

    df = pd.DataFrame({'apple': string_var, 'ban': string_var})
    design = tools.categorical(df, 'apple', drop=True)
    assert_frame_equal(design, dummies)


def test_categorical_errors(string_var):
    with pytest.raises(ValueError, match='Can only convert one column'):
        tools.categorical(string_var, (0, 1))
    with pytest.raises(ValueError, match='data.name does not match col'):
        tools.categorical(string_var, {'a': 1})
