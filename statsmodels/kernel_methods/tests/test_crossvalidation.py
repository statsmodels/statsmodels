from __future__ import division, absolute_import, print_function

import pytest

from . import kde_utils
from .. import kde
from .. import bandwidths

parameters = kde_utils.createParams_norm()

class TestCV(object):
    def createKDE(self, data, method, **args):
        all_args = dict(parameters.args)
        all_args.update(args)
        k = kde.KDE(data, **all_args)
        if method.instance is None:
            del k.method
        else:
            k.method = method.instance
        if method.bound_low:
            k.lower = parameters.lower
        else:
            del k.lower
        if method.bound_high:
            k.upper = parameters.upper
        else:
            del k.upper
        return k

    @pytest.mark.parametrize('method', parameters.methods)
    def test_loo(self, method):
        k = self.createKDE(parameters.vs[0], method)
        k.bandwidth = bandwidths.CrossValidation()
        est = k.fit()
        assert est.bandwidth > 0

    @pytest.mark.parametrize('method', parameters.methods)
    def test_folds(self, method ):
        k = self.createKDE(parameters.vs[0], method)
        imse_args = dict(use_grid=True, folding=5)
        k.bandwidth = bandwidths.CrossValidation(**imse_args)
        est = k.fit()
        assert est.bandwidth > 0

    @pytest.mark.parametrize('method', parameters.methods)
    def test_imse(self, method):
        k = self.createKDE(parameters.vs[0], method)
        imse_args = dict(use_grid=True, folding=5)
        k.bandwidth = bandwidths.CrossValidation(bandwidths.CVIMSE, **imse_args)
        est = k.fit()
        assert est.bandwidth > 0

    @pytest.mark.parametrize('method', parameters.methods)
    def test_sampling(self, method):
        k = self.createKDE(parameters.vs[0], method)
        imse_args = dict(sampling=100)
        k.bandwidth = bandwidths.CrossValidation(**imse_args)
        est = k.fit()
        assert est.bandwidth > 0

