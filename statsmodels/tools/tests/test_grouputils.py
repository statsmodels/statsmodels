import numpy as np
import pandas as pd
from statsmodels.tools.grouputils import Grouping
from statsmodels.tools.tools import categorical
from statsmodels.datasets import grunfeld, anes96
from pandas.util import testing as ptesting


class CheckGrouping(object):

    def test_reindex(self):
        # smoke test
        self.grouping.reindex(self.grouping.index)

    def test_count_categories(self):
        self.grouping.count_categories(level=0)
        np.testing.assert_equal(self.grouping.counts, self.expected_counts)

    def test_sort(self):
        # data frame
        sorted_data, index = self.grouping.sort(self.data)
        expected_sorted_data = self.data.sort_index()

        ptesting.assert_frame_equal(sorted_data, expected_sorted_data)
        np.testing.assert_(isinstance(sorted_data, pd.DataFrame))
        np.testing.assert_(not index.equals(self.grouping.index))

        # make sure it copied
        if hasattr(sorted_data, 'equals'): # newer pandas
            np.testing.assert_(not sorted_data.equals(self.data))

        # 2d arrays
        sorted_data, index = self.grouping.sort(self.data.values)
        np.testing.assert_array_equal(sorted_data,
                                      expected_sorted_data.values)
        np.testing.assert_(isinstance(sorted_data, np.ndarray))

        # 1d series
        series = self.data[self.data.columns[0]]
        sorted_data, index = self.grouping.sort(series)

        expected_sorted_data = series.sort_index()
        ptesting.assert_series_equal(sorted_data, expected_sorted_data)
        np.testing.assert_(isinstance(sorted_data, pd.Series))
        if hasattr(sorted_data, 'equals'):
            np.testing.assert_(not sorted_data.equals(series))

        # 1d array
        array = series.values
        sorted_data, index = self.grouping.sort(array)

        expected_sorted_data = series.sort_index().values
        np.testing.assert_array_equal(sorted_data, expected_sorted_data)
        np.testing.assert_(isinstance(sorted_data, np.ndarray))

    def test_transform_dataframe(self):
        names = self.data.index.names
        transformed_dataframe = self.grouping.transform_dataframe(
                                            self.data,
                                            lambda x : x.mean(),
                                            level=0)
        expected = self.data.reset_index().groupby(names[0]
                                            ).apply(lambda x : x.mean())[
                                                    self.data.columns]
        np.testing.assert_array_equal(transformed_dataframe,
                                      expected.values)

        if len(names) > 1:
            transformed_dataframe = self.grouping.transform_dataframe(
                                            self.data, lambda x : x.mean(),
                                            level=1)
            expected = self.data.reset_index().groupby(names[1]
                                                      ).apply(lambda x :
                                                              x.mean())[
                                                        self.data.columns]
            np.testing.assert_array_equal(transformed_dataframe,
                                          expected.values)

    def test_transform_array(self):
        names = self.data.index.names
        transformed_array = self.grouping.transform_array(
                                            self.data.values,
                                            lambda x : x.mean(),
                                            level=0)
        expected = self.data.reset_index().groupby(names[0]
                                            ).apply(lambda x : x.mean())[
                                                    self.data.columns]
        np.testing.assert_array_equal(transformed_array,
                                      expected.values)

        if len(names) > 1:
            transformed_array = self.grouping.transform_array(
                                            self.data.values,
                                            lambda x : x.mean(), level=1)
            expected = self.data.reset_index().groupby(names[1]
                                                      ).apply(lambda x :
                                                              x.mean())[
                                                        self.data.columns]
            np.testing.assert_array_equal(transformed_array,
                                          expected.values)


    def test_transform_slices(self):
        names = self.data.index.names
        transformed_slices = self.grouping.transform_slices(
                                            self.data.values,
                                            lambda x, idx : x.mean(0),
                                            level=0)
        expected = self.data.reset_index().groupby(names[0]).mean()[
                                                    self.data.columns]
        np.testing.assert_allclose(transformed_slices, expected.values,
                                   rtol=1e-12, atol=1e-25)

        if len(names) > 1:
            transformed_slices = self.grouping.transform_slices(
                                            self.data.values,
                                            lambda x, idx : x.mean(0),
                                            level=1)
            expected = self.data.reset_index().groupby(names[1]
                                                       ).mean()[
                                                        self.data.columns]
            np.testing.assert_allclose(transformed_slices, expected.values,
                                       rtol=1e-12, atol=1e-25)

    def test_dummies_groups(self):
        # smoke test, calls dummy_sparse under the hood
        self.grouping.dummies_groups()

        if len(self.grouping.group_names) > 1:
            self.grouping.dummies_groups(level=1)

    def test_dummy_sparse(self):
        data = self.data
        self.grouping.dummy_sparse()
        expected = categorical(data.index.get_level_values(0).values,
                               drop=True)
        np.testing.assert_equal(self.grouping._dummies.toarray(), expected)

        if len(self.grouping.group_names) > 1:
            self.grouping.dummy_sparse(level=1)
            expected = categorical(data.index.get_level_values(1).values,
                    drop=True)
            np.testing.assert_equal(self.grouping._dummies.toarray(),
                                    expected)


class TestMultiIndexGrouping(CheckGrouping):
    @classmethod
    def setup_class(cls):
        grun_data = grunfeld.load_pandas().data
        multi_index_data = grun_data.set_index(['firm', 'year'])
        multi_index_panel = multi_index_data.index
        cls.grouping = Grouping(multi_index_panel)
        cls.data = multi_index_data

        cls.expected_counts = [20] * 11


class TestIndexGrouping(CheckGrouping):
    @classmethod
    def setup_class(cls):
        grun_data = grunfeld.load_pandas().data
        index_data = grun_data.set_index(['firm'])
        index_group = index_data.index
        cls.grouping = Grouping(index_group)
        cls.data = index_data

        cls.expected_counts = [20] * 11


def test_init_api():
    # make a multi-index panel
    grun_data = grunfeld.load_pandas().data
    multi_index_panel = grun_data.set_index(['firm', 'year']).index
    grouping = Grouping(multi_index_panel)
    # check group_names
    np.testing.assert_array_equal(grouping.group_names, ['firm', 'year'])
    # check shape
    np.testing.assert_array_equal(grouping.index_shape, (11, 20))
    # check index_int
    np.testing.assert_array_equal(grouping.labels,
      [[ 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
         5, 5, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
         8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
         4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 7,
         7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
         7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
         9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
         10, 10, 10, 10, 10, 10, 10, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
         6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3, 3, 3, 3, 3,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
         17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
         14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
         11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7,
         8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4,
         5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1,
         2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
         19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
         16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
         13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
         10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6,
         7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3,
         4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
    grouping = Grouping(multi_index_panel, names=['firms', 'year'])
    np.testing.assert_array_equal(grouping.group_names, ['firms', 'year'])

    # make a multi-index grouping
    anes_data = anes96.load_pandas().data
    multi_index_groups = anes_data.set_index(['educ', 'income',
                                              'TVnews']).index
    grouping = Grouping(multi_index_groups)
    np.testing.assert_array_equal(grouping.group_names,
                                  ['educ', 'income', 'TVnews'])
    np.testing.assert_array_equal(grouping.index_shape, (7, 24, 8))

    # make a list multi-index panel
    list_panel = multi_index_panel.tolist()
    grouping = Grouping(list_panel, names=['firms', 'year'])
    np.testing.assert_array_equal(grouping.group_names, ['firms', 'year'])
    np.testing.assert_array_equal(grouping.index_shape, (11, 20))

    # make a list multi-index grouping
    list_groups = multi_index_groups.tolist()
    grouping = Grouping(list_groups, names=['educ', 'income', 'TVnews'])
    np.testing.assert_array_equal(grouping.group_names,
                                  ['educ', 'income', 'TVnews'])
    np.testing.assert_array_equal(grouping.index_shape, (7, 24, 8))


    # single-variable index grouping
    index_group = multi_index_panel.get_level_values(0)
    grouping = Grouping(index_group)
    # the original multi_index_panel had it's name changed inplace above
    np.testing.assert_array_equal(grouping.group_names, ['firms'])
    np.testing.assert_array_equal(grouping.index_shape, (220,))

    # single variable list grouping
    list_group = multi_index_panel.get_level_values(0).tolist()
    grouping = Grouping(list_group)
    np.testing.assert_array_equal(grouping.group_names, ["group0"])
    np.testing.assert_array_equal(grouping.index_shape, 11*20)

    # test generic group names
    grouping = Grouping(list_groups)
    np.testing.assert_array_equal(grouping.group_names,
                                  ['group0', 'group1', 'group2'])
