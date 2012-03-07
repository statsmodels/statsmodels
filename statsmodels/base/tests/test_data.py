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
        np.testing.assert_equal(self.data._orig_endog, self.endog)
        np.testing.assert_equal(self.data._orig_exog, self.exog)

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
        np.testing.assert_equal(self.data.row_labels, self.row_labels)


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
        np.testing.assert_equal(self.data._orig_endog, self.endog)
        np.testing.assert_equal(self.data._orig_exog, self.exog.squeeze())


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
        ptesting.assert_frame_equal(self.data._orig_endog, self.endog)
        ptesting.assert_frame_equal(self.data._orig_exog, self.exog)

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
        np.testing.assert_equal(self.data._orig_endog, self.endog)
        ptesting.assert_frame_equal(self.data._orig_exog, self.exog)


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
        ptesting.assert_frame_equal(self.data._orig_endog, self.endog)
        np.testing.assert_equal(self.data._orig_exog, self.exog)


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
        np.testing.assert_equal(self.data._orig_endog, self.endog)
        ptesting.assert_frame_equal(self.data._orig_exog, self.exog)


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
        ptesting.assert_frame_equal(self.data._orig_endog, self.endog)
        np.testing.assert_equal(self.data._orig_exog, self.exog)


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
        ptesting.assert_series_equal(self.data._orig_endog, self.endog)
        ptesting.assert_frame_equal(self.data._orig_exog, self.exog)


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
        ptesting.assert_series_equal(self.data._orig_endog, self.endog)
        ptesting.assert_series_equal(self.data._orig_exog, self.exog)

    def test_endogexog(self):
        np.testing.assert_equal(self.data.endog, self.endog.values.squeeze())
        np.testing.assert_equal(self.data.exog, self.exog.values[:,None])


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

if __name__ == "__main__":
    import nose
    #nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
    #        exit=False)
    nose.runmodule(argv=[__file__, '-vvs', '-x'], exit=False)
