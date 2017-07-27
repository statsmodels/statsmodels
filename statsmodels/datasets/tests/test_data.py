from unittest import TestCase
from nose.tools import assert_true

import numpy as np
import pandas as pd

import statsmodels.datasets as datasets
from statsmodels.datasets import co2
from statsmodels.datasets.utils import Dataset


def test_co2_python3():
    # this failed in pd.to_datetime on Python 3 with pandas <= 0.12.0
    dta = co2.load_pandas()


class TestDatasets(object):

    @classmethod
    def setup_class(cls):
        exclude = ['check_internet', 'clear_data_home', 'get_data_home',
                   'get_rdataset', 'tests', 'utils', 'webuse']
        cls.sets = []
        for dataset_name in dir(datasets):
            if not dataset_name.startswith('_') and dataset_name not in exclude:
                cls.sets.append(dataset_name)

    def check(self, dataset_name):
        dataset = __import__('statsmodels.datasets.' + dataset_name, fromlist=[''])
        data = dataset.load()
        assert_true(isinstance(data, Dataset))
        assert_true(isinstance(data.data, np.recarray))

        df_data = dataset.load_pandas()
        assert_true(isinstance(data, Dataset))
        assert_true(isinstance(df_data.data, pd.DataFrame))

    def test_all_datasets(self):
        for dataset_name in self.sets:
            yield (self.check, dataset_name)
