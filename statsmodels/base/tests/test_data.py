import numpy as np
import pandas
import pandas.util.testing as ptesting

from statsmodels.base import data as sm_data

#class TestDates(object):
#    @classmethod
#    def setupClass(cls):
#        nrows = 10
#        cls.dates_result = cls.dates_results = np.random.random(nrows)
#
#    def test_dates(self):
#        np.testing.assert_equal(data.wrap_output(self.dates_input, 'dates'),
#                                self.dates_result)

class TestArrays(object):
    @classmethod
    def setupClass(cls):
        cls.endog = np.random.random(10)
        cls.exog = np.c_[np.ones(10), np.random.random((10,2))]
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        nrows = 10
        nvars = 3
        cls.col_result = cls.col_input = np.random.random(nvars)
        cls.row_result = cls.row_input = np.random.random(nrows)
        cls.cov_result = cls.cov_input = np.random.random((nvars, nvars))
        cls.xnames = ['const', 'x1', 'x2']
        cls.ynames = 'y'
        cls.row_labels = None

    def test_orig(self):
        np.testing.assert_equal(self.data.orig_endog, self.endog)
        np.testing.assert_equal(self.data.orig_exog, self.exog)

    def test_endogexog(self):
        np.testing.assert_equal(self.data.endog, self.endog)
        np.testing.assert_equal(self.data.exog, self.exog)

    def test_attach(self):
        data = self.data
        # this makes sure what the wrappers need work but not the wrapped
        # results themselves
        np.testing.assert_equal(data.wrap_output(self.col_input, 'columns'),
                                self.col_result)
        np.testing.assert_equal(data.wrap_output(self.row_input, 'rows'),
                                self.row_result)
        np.testing.assert_equal(data.wrap_output(self.cov_input, 'cov'),
                                self.cov_result)

    def test_names(self):
        data = self.data
        np.testing.assert_equal(data.xnames, self.xnames)
        np.testing.assert_equal(data.ynames, self.ynames)

    def test_labels(self):
        #HACK: because numpy master after NA stuff assert_equal fails on
        # pandas indices
        np.testing.assert_(np.all(self.data.row_labels == self.row_labels))


class TestArrays2dEndog(TestArrays):
    @classmethod
    def setupClass(cls):
        super(TestArrays2dEndog, cls).setupClass()
        cls.endog = np.random.random((10,1))
        cls.exog = np.c_[np.ones(10), np.random.random((10,2))]
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        #cls.endog = endog.squeeze()

    def test_endogexog(self):
        np.testing.assert_equal(self.data.endog, self.endog.squeeze())
        np.testing.assert_equal(self.data.exog, self.exog)


class TestArrays1dExog(TestArrays):
    @classmethod
    def setupClass(cls):
        super(TestArrays1dExog, cls).setupClass()
        cls.endog = np.random.random(10)
        exog =  np.random.random(10)
        cls.data = sm_data.handle_data(cls.endog, exog)
        cls.exog = exog[:,None]
        cls.xnames = ['x1']
        cls.ynames = 'y'

    def test_orig(self):
        np.testing.assert_equal(self.data.orig_endog, self.endog)
        np.testing.assert_equal(self.data.orig_exog, self.exog.squeeze())


class TestDataFrames(TestArrays):
    @classmethod
    def setupClass(cls):
        cls.endog = pandas.DataFrame(np.random.random(10), columns=['y_1'])
        exog =  pandas.DataFrame(np.random.random((10,2)),
                                 columns=['x_1','x_2'])
        exog.insert(0, 'const', 1)
        cls.exog = exog
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        nrows = 10
        nvars = 3
        cls.col_input = np.random.random(nvars)
        cls.col_result = pandas.Series(cls.col_input,
                                          index=exog.columns)
        cls.row_input = np.random.random(nrows)
        cls.row_result = pandas.Series(cls.row_input,
                                          index=exog.index)
        cls.cov_input = np.random.random((nvars, nvars))
        cls.cov_result = pandas.DataFrame(cls.cov_input,
                                           index = exog.columns,
                                           columns = exog.columns)
        cls.xnames = ['const', 'x_1', 'x_2']
        cls.ynames = 'y_1'
        cls.row_labels = cls.exog.index

    def test_orig(self):
        ptesting.assert_frame_equal(self.data.orig_endog, self.endog)
        ptesting.assert_frame_equal(self.data.orig_exog, self.exog)

    def test_endogexog(self):
        np.testing.assert_equal(self.data.endog, self.endog.values.squeeze())
        np.testing.assert_equal(self.data.exog, self.exog.values)

    def test_attach(self):
        data = self.data
        # this makes sure what the wrappers need work but not the wrapped
        # results themselves
        ptesting.assert_series_equal(data.wrap_output(self.col_input,
                                                  'columns'),
                                self.col_result)
        ptesting.assert_series_equal(data.wrap_output(self.row_input, 'rows'),
                                self.row_result)
        ptesting.assert_frame_equal(data.wrap_output(self.cov_input, 'cov'),
                                self.cov_result)


class TestLists(TestArrays):
    @classmethod
    def setupClass(cls):
        super(TestLists, cls).setupClass()
        cls.endog = np.random.random(10).tolist()
        cls.exog = np.c_[np.ones(10), np.random.random((10,2))].tolist()
        cls.data = sm_data.handle_data(cls.endog, cls.exog)


class TestRecarrays(TestArrays):
    @classmethod
    def setupClass(cls):
        super(TestRecarrays, cls).setupClass()
        cls.endog = np.random.random(9).view([('y_1',
                                         'f8')]).view(np.recarray)
        exog = np.random.random(9*3).view([('const', 'f8'),('x_1', 'f8'),
                                ('x_2', 'f8')]).view(np.recarray)
        exog['const'] = 1
        cls.exog = exog
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        cls.xnames = ['const', 'x_1', 'x_2']
        cls.ynames = 'y_1'

    def test_endogexog(self):
        np.testing.assert_equal(self.data.endog, self.endog.view(float))
        np.testing.assert_equal(self.data.exog, self.exog.view((float,3)))


class TestStructarrays(TestArrays):
    @classmethod
    def setupClass(cls):
        super(TestStructarrays, cls).setupClass()
        cls.endog = np.random.random(9).view([('y_1',
                                         'f8')]).view(np.recarray)
        exog = np.random.random(9*3).view([('const', 'f8'),('x_1', 'f8'),
                                ('x_2', 'f8')]).view(np.recarray)
        exog['const'] = 1
        cls.exog = exog
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        cls.xnames = ['const', 'x_1', 'x_2']
        cls.ynames = 'y_1'

    def test_endogexog(self):
        np.testing.assert_equal(self.data.endog, self.endog.view(float))
        np.testing.assert_equal(self.data.exog, self.exog.view((float,3)))


class TestListDataFrame(TestDataFrames):
    @classmethod
    def setupClass(cls):
        cls.endog = np.random.random(10).tolist()

        exog =  pandas.DataFrame(np.random.random((10,2)),
                                 columns=['x_1','x_2'])
        exog.insert(0, 'const', 1)
        cls.exog = exog
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        nrows = 10
        nvars = 3
        cls.col_input = np.random.random(nvars)
        cls.col_result = pandas.Series(cls.col_input,
                                          index=exog.columns)
        cls.row_input = np.random.random(nrows)
        cls.row_result = pandas.Series(cls.row_input,
                                          index=exog.index)
        cls.cov_input = np.random.random((nvars, nvars))
        cls.cov_result = pandas.DataFrame(cls.cov_input,
                                           index = exog.columns,
                                           columns = exog.columns)
        cls.xnames = ['const', 'x_1', 'x_2']
        cls.ynames = 'y'
        cls.row_labels = cls.exog.index

    def test_endogexog(self):
        np.testing.assert_equal(self.data.endog, self.endog)
        np.testing.assert_equal(self.data.exog, self.exog.values)

    def test_orig(self):
        np.testing.assert_equal(self.data.orig_endog, self.endog)
        ptesting.assert_frame_equal(self.data.orig_exog, self.exog)


class TestDataFrameList(TestDataFrames):
    @classmethod
    def setupClass(cls):
        cls.endog = pandas.DataFrame(np.random.random(10), columns=['y_1'])

        exog =  pandas.DataFrame(np.random.random((10,2)),
                                 columns=['x1','x2'])
        exog.insert(0, 'const', 1)
        cls.exog = exog.values.tolist()
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        nrows = 10
        nvars = 3
        cls.col_input = np.random.random(nvars)
        cls.col_result = pandas.Series(cls.col_input,
                                          index=exog.columns)
        cls.row_input = np.random.random(nrows)
        cls.row_result = pandas.Series(cls.row_input,
                                          index=exog.index)
        cls.cov_input = np.random.random((nvars, nvars))
        cls.cov_result = pandas.DataFrame(cls.cov_input,
                                           index = exog.columns,
                                           columns = exog.columns)
        cls.xnames = ['const', 'x1', 'x2']
        cls.ynames = 'y_1'
        cls.row_labels = cls.endog.index

    def test_endogexog(self):
        np.testing.assert_equal(self.data.endog, self.endog.values.squeeze())
        np.testing.assert_equal(self.data.exog, self.exog)

    def test_orig(self):
        ptesting.assert_frame_equal(self.data.orig_endog, self.endog)
        np.testing.assert_equal(self.data.orig_exog, self.exog)


class TestArrayDataFrame(TestDataFrames):
    @classmethod
    def setupClass(cls):
        cls.endog = np.random.random(10)

        exog =  pandas.DataFrame(np.random.random((10,2)),
                                 columns=['x_1','x_2'])
        exog.insert(0, 'const', 1)
        cls.exog = exog
        cls.data = sm_data.handle_data(cls.endog, exog)
        nrows = 10
        nvars = 3
        cls.col_input = np.random.random(nvars)
        cls.col_result = pandas.Series(cls.col_input,
                                          index=exog.columns)
        cls.row_input = np.random.random(nrows)
        cls.row_result = pandas.Series(cls.row_input,
                                          index=exog.index)
        cls.cov_input = np.random.random((nvars, nvars))
        cls.cov_result = pandas.DataFrame(cls.cov_input,
                                           index = exog.columns,
                                           columns = exog.columns)
        cls.xnames = ['const', 'x_1', 'x_2']
        cls.ynames = 'y'
        cls.row_labels = cls.exog.index

    def test_endogexog(self):
        np.testing.assert_equal(self.data.endog, self.endog)
        np.testing.assert_equal(self.data.exog, self.exog.values)

    def test_orig(self):
        np.testing.assert_equal(self.data.orig_endog, self.endog)
        ptesting.assert_frame_equal(self.data.orig_exog, self.exog)


class TestDataFrameArray(TestDataFrames):
    @classmethod
    def setupClass(cls):
        cls.endog = pandas.DataFrame(np.random.random(10), columns=['y_1'])

        exog =  pandas.DataFrame(np.random.random((10,2)),
                                 columns=['x1','x2']) # names mimic defaults
        exog.insert(0, 'const', 1)
        cls.exog = exog.values
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        nrows = 10
        nvars = 3
        cls.col_input = np.random.random(nvars)
        cls.col_result = pandas.Series(cls.col_input,
                                          index=exog.columns)
        cls.row_input = np.random.random(nrows)
        cls.row_result = pandas.Series(cls.row_input,
                                          index=exog.index)
        cls.cov_input = np.random.random((nvars, nvars))
        cls.cov_result = pandas.DataFrame(cls.cov_input,
                                           index = exog.columns,
                                           columns = exog.columns)
        cls.xnames = ['const', 'x1', 'x2']
        cls.ynames = 'y_1'
        cls.row_labels = cls.endog.index

    def test_endogexog(self):
        np.testing.assert_equal(self.data.endog, self.endog.values.squeeze())
        np.testing.assert_equal(self.data.exog, self.exog)

    def test_orig(self):
        ptesting.assert_frame_equal(self.data.orig_endog, self.endog)
        np.testing.assert_equal(self.data.orig_exog, self.exog)


class TestSeriesDataFrame(TestDataFrames):
    @classmethod
    def setupClass(cls):
        cls.endog = pandas.Series(np.random.random(10), name='y_1')

        exog =  pandas.DataFrame(np.random.random((10,2)),
                                 columns=['x_1','x_2'])
        exog.insert(0, 'const', 1)
        cls.exog = exog
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        nrows = 10
        nvars = 3
        cls.col_input = np.random.random(nvars)
        cls.col_result = pandas.Series(cls.col_input,
                                          index=exog.columns)
        cls.row_input = np.random.random(nrows)
        cls.row_result = pandas.Series(cls.row_input,
                                          index=exog.index)
        cls.cov_input = np.random.random((nvars, nvars))
        cls.cov_result = pandas.DataFrame(cls.cov_input,
                                           index = exog.columns,
                                           columns = exog.columns)
        cls.xnames = ['const', 'x_1', 'x_2']
        cls.ynames = 'y_1'
        cls.row_labels = cls.exog.index

    def test_orig(self):
        ptesting.assert_series_equal(self.data.orig_endog, self.endog)
        ptesting.assert_frame_equal(self.data.orig_exog, self.exog)


class TestSeriesSeries(TestDataFrames):
    @classmethod
    def setupClass(cls):
        cls.endog = pandas.Series(np.random.random(10), name='y_1')

        exog =  pandas.Series(np.random.random(10), name='x_1')
        cls.exog = exog
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        nrows = 10
        nvars = 1
        cls.col_input = np.random.random(nvars)
        cls.col_result = pandas.Series(cls.col_input,
                                          index = [exog.name])
        cls.row_input = np.random.random(nrows)
        cls.row_result = pandas.Series(cls.row_input,
                                          index = exog.index)
        cls.cov_input = np.random.random((nvars, nvars))
        cls.cov_result = pandas.DataFrame(cls.cov_input,
                                           index = [exog.name],
                                           columns = [exog.name])
        cls.xnames = ['x_1']
        cls.ynames = 'y_1'
        cls.row_labels = cls.exog.index

    def test_orig(self):
        ptesting.assert_series_equal(self.data.orig_endog, self.endog)
        ptesting.assert_series_equal(self.data.orig_exog, self.exog)

    def test_endogexog(self):
        np.testing.assert_equal(self.data.endog, self.endog.values.squeeze())
        np.testing.assert_equal(self.data.exog, self.exog.values[:,None])

def test_alignment():
    """
    Fix Issue #206
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.datasets.macrodata import load_pandas

    d = load_pandas().data
    #growth rates
    gs_l_realinv = 400 * np.log(d['realinv']).diff().dropna()
    gs_l_realgdp = 400 * np.log(d['realgdp']).diff().dropna()
    lint = d['realint'][:-1] # incorrect indexing for test purposes

    endog = gs_l_realinv

    # re-index because they won't conform to lint
    realgdp = gs_l_realgdp.reindex(lint.index, method='bfill')
    data = dict(const=np.ones_like(lint), lrealgdp=realgdp, lint=lint)
    exog = pandas.DataFrame(data)

    # which index do we get??
    np.testing.assert_raises(ValueError, OLS, *(endog, exog))

class TestMultipleEqsArrays(TestArrays):
    @classmethod
    def setupClass(cls):
        cls.endog = np.random.random((10,4))
        cls.exog = np.c_[np.ones(10), np.random.random((10,2))]
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        nrows = 10
        nvars = 3
        neqs = 4
        cls.col_result = cls.col_input = np.random.random(nvars)
        cls.row_result = cls.row_input = np.random.random(nrows)
        cls.cov_result = cls.cov_input = np.random.random((nvars, nvars))
        cls.cov_eq_result = cls.cov_eq_input = np.random.random((neqs,neqs))
        cls.col_eq_result = cls.col_eq_input = np.array((neqs, nvars))
        cls.xnames = ['const', 'x1', 'x2']
        cls.ynames = ['y1', 'y2', 'y3', 'y4']
        cls.row_labels = None

    def test_attach(self):
        data = self.data
        # this makes sure what the wrappers need work but not the wrapped
        # results themselves
        np.testing.assert_equal(data.wrap_output(self.col_input, 'columns'),
                                self.col_result)
        np.testing.assert_equal(data.wrap_output(self.row_input, 'rows'),
                                self.row_result)
        np.testing.assert_equal(data.wrap_output(self.cov_input, 'cov'),
                                self.cov_result)
        np.testing.assert_equal(data.wrap_output(self.cov_eq_input, 'cov_eq'),
                                self.cov_eq_result)
        np.testing.assert_equal(data.wrap_output(self.col_eq_input,
                                                 'columns_eq'),
                                self.col_eq_result)


class TestMultipleEqsDataFrames(TestDataFrames):
    @classmethod
    def setupClass(cls):
        cls.endog = endog = pandas.DataFrame(np.random.random((10,4)),
                                     columns=['y_1', 'y_2', 'y_3', 'y_4'])
        exog =  pandas.DataFrame(np.random.random((10,2)),
                                 columns=['x_1','x_2'])
        exog.insert(0, 'const', 1)
        cls.exog = exog
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        nrows = 10
        nvars = 3
        neqs = 4
        cls.col_input = np.random.random(nvars)
        cls.col_result = pandas.Series(cls.col_input,
                                          index=exog.columns)
        cls.row_input = np.random.random(nrows)
        cls.row_result = pandas.Series(cls.row_input,
                                          index=exog.index)
        cls.cov_input = np.random.random((nvars, nvars))
        cls.cov_result = pandas.DataFrame(cls.cov_input,
                                           index = exog.columns,
                                           columns = exog.columns)
        cls.cov_eq_input = np.random.random((neqs, neqs))
        cls.cov_eq_result = pandas.DataFrame(cls.cov_eq_input,
                                              index=endog.columns,
                                              columns=endog.columns)
        cls.col_eq_input = np.random.random((nvars, neqs))
        cls.col_eq_result = pandas.DataFrame(cls.col_eq_input,
                                              index=exog.columns,
                                              columns=endog.columns)
        cls.xnames = ['const', 'x_1', 'x_2']
        cls.ynames = ['y_1', 'y_2', 'y_3', 'y_4']
        cls.row_labels = cls.exog.index

    def test_attach(self):
        data = self.data
        ptesting.assert_series_equal(data.wrap_output(self.col_input,
                                                  'columns'),
                                self.col_result)
        ptesting.assert_series_equal(data.wrap_output(self.row_input, 'rows'),
                                self.row_result)
        ptesting.assert_frame_equal(data.wrap_output(self.cov_input, 'cov'),
                                self.cov_result)
        ptesting.assert_frame_equal(data.wrap_output(self.cov_eq_input,
                                    'cov_eq'),
                                self.cov_eq_result)
        ptesting.assert_frame_equal(data.wrap_output(self.col_eq_input,
                                                 'columns_eq'),
                                self.col_eq_result)

class TestMissingArray(object):
    @classmethod
    def setupClass(cls):
        X = np.random.random((25,4))
        y = np.random.random(25)
        y[10] = np.nan
        X[2,3] = np.nan
        X[14,2] = np.nan
        cls.y, cls.X = y, X

    def test_raise(self):
        np.testing.assert_raises(Exception, sm_data.handle_data,
                                            (self.y, self.X, 'raise'))

    def test_drop(self):
        y = self.y
        X = self.X
        combined = np.c_[y, X]
        idx = ~np.isnan(combined).any(axis=1)
        y = y[idx]
        X = X[idx]
        data = sm_data.handle_data(self.y, self.X, 'drop')
        np.testing.assert_array_equal(data.endog, y)
        np.testing.assert_array_equal(data.exog, X)

    def test_none(self):
        data = sm_data.handle_data(self.y, self.X, 'none')
        np.testing.assert_array_equal(data.endog, self.y)
        np.testing.assert_array_equal(data.exog, self.X)

    def test_endog_only_raise(self):
        np.testing.assert_raises(Exception, sm_data.handle_data,
                                            (self.y, None, 'raise'))

    def test_endog_only_drop(self):
        y = self.y
        y = y[~np.isnan(y)]
        data = sm_data.handle_data(self.y, None, 'drop')
        np.testing.assert_array_equal(data.endog, y)

    def test_mv_endog(self):
        y = self.X
        y = y[~np.isnan(y).any(axis=1)]
        data = sm_data.handle_data(self.X, None, 'drop')
        np.testing.assert_array_equal(data.endog, y)

    def test_extra_kwargs_2d(self):
        sigma = np.random.random((25, 25))
        sigma = sigma + sigma.T - np.diag(np.diag(sigma))
        data = sm_data.handle_data(self.y, self.X, 'drop', sigma=sigma)
        idx = ~np.isnan(np.c_[self.y, self.X]).any(axis=1)
        sigma = sigma[idx][:,idx]
        np.testing.assert_array_equal(data.sigma, sigma)

    def test_extra_kwargs_1d(self):
        weights = np.random.random(25)
        data = sm_data.handle_data(self.y, self.X, 'drop', weights=weights)
        idx = ~np.isnan(np.c_[self.y, self.X]).any(axis=1)
        weights = weights[idx]
        np.testing.assert_array_equal(data.weights, weights)

class TestMissingPandas(object):
    @classmethod
    def setupClass(cls):
        X = np.random.random((25,4))
        y = np.random.random(25)
        y[10] = np.nan
        X[2,3] = np.nan
        X[14,2] = np.nan
        cls.y, cls.X = pandas.Series(y), pandas.DataFrame(X)

    def test_raise(self):
        np.testing.assert_raises(Exception, sm_data.handle_data,
                                            (self.y, self.X, 'raise'))

    def test_drop(self):
        y = self.y
        X = self.X
        combined = np.c_[y, X]
        idx = ~np.isnan(combined).any(axis=1)
        y = y.ix[idx]
        X = X.ix[idx]
        data = sm_data.handle_data(self.y, self.X, 'drop')
        np.testing.assert_array_equal(data.endog, y.values)
        ptesting.assert_series_equal(data.orig_endog, self.y.ix[idx])
        np.testing.assert_array_equal(data.exog, X.values)
        ptesting.assert_frame_equal(data.orig_exog, self.X.ix[idx])

    def test_none(self):
        data = sm_data.handle_data(self.y, self.X, 'none')
        np.testing.assert_array_equal(data.endog, self.y.values)
        np.testing.assert_array_equal(data.exog, self.X.values)

    def test_endog_only_raise(self):
        np.testing.assert_raises(Exception, sm_data.handle_data,
                                            (self.y, None, 'raise'))

    def test_endog_only_drop(self):
        y = self.y
        y = y.dropna()
        data = sm_data.handle_data(self.y, None, 'drop')
        np.testing.assert_array_equal(data.endog, y.values)

    def test_mv_endog(self):
        y = self.X
        y = y.ix[~np.isnan(y.values).any(axis=1)]
        data = sm_data.handle_data(self.X, None, 'drop')
        np.testing.assert_array_equal(data.endog, y.values)

    def test_labels(self):
        2, 10, 14
        labels = pandas.Index([0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15,
                               16, 17, 18, 19, 20, 21, 22, 23, 24])
        data = sm_data.handle_data(self.y, self.X, 'drop')
        np.testing.assert_(data.row_labels.equals(labels))



if __name__ == "__main__":
    import nose
    #nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
    #        exit=False)
    nose.runmodule(argv=[__file__, '-vvs', '-x'], exit=False)
