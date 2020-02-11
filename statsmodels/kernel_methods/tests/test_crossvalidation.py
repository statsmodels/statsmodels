import pytest

from .. import bandwidths
from .kde_test_utils import kde_tester, datasets, generate_methods_data, kde_tester_args

all_methods_small_data = generate_methods_data(['norm'], indices=[0])

@pytest.mark.parametrize(kde_tester_args, all_methods_small_data)
class TestCV(object):
    @kde_tester
    def test_loo(self, k, method, data):
        k.bandwidth = bandwidths.CrossValidation()
        est = k.fit()
        assert est.bandwidth > 0

    @kde_tester
    def test_folds(self, k, method, data):
        imse_args = dict(use_grid=True, folding=5)
        k.bandwidth = bandwidths.CrossValidation(**imse_args)
        est = k.fit()
        assert est.bandwidth > 0

    @kde_tester
    def test_imse(self, k, method, data):
        imse_args = dict(use_grid=True, folding=5)
        k.bandwidth = bandwidths.CrossValidation(bandwidths.CVIMSE, **imse_args)
        est = k.fit()
        assert est.bandwidth > 0

    @kde_tester
    def test_sampling(self, k, method, data):
        imse_args = dict(sampling=100)
        k.bandwidth = bandwidths.CrossValidation(**imse_args)
        est = k.fit()
        assert est.bandwidth > 0
