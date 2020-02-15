import pytest

from .. import bandwidths
from .kde_datasets import DataSets, createKDE

all_methods_small_data = DataSets.norm([128])


@pytest.mark.parametrize('data', all_methods_small_data)
class TestCV(object):
    def test_loo(self, data):
        k = createKDE(data)
        k.bandwidth = bandwidths.CrossValidation()
        est = k.fit()
        assert est.bandwidth > 0

    def test_folds(self, data):
        k = createKDE(data)
        imse_args = dict(use_grid=True, folding=5)
        k.bandwidth = bandwidths.CrossValidation(**imse_args)
        est = k.fit()
        assert est.bandwidth > 0

    def test_imse(self, data):
        k = createKDE(data)
        imse_args = dict(use_grid=True, folding=5)
        k.bandwidth = bandwidths.CrossValidation(bandwidths.CVIMSE,
                                                 **imse_args)
        est = k.fit()
        assert est.bandwidth > 0

    def test_sampling(self, data):
        k = createKDE(data)
        imse_args = dict(sampling=100)
        k.bandwidth = bandwidths.CrossValidation(bandwidths.CVIMSE,
                                                 **imse_args)
        est = k.fit()
        assert est.bandwidth > 0
