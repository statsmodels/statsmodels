from __future__ import division, absolute_import, print_function

from . import kde_utils
from .. import kde
from .. import bandwidths

class TestCV(object):
    def createKDE(self, data, method, **args):
        all_args = dict(self.args)
        all_args.update(args)
        k = kde.KDE(data, **all_args)
        if method.instance is None:
            del k.method
        else:
            k.method = method.instance
        if method.bound_low:
            k.lower = self.lower
        else:
            del k.lower
        if method.bound_high:
            k.upper = self.upper
        else:
            del k.upper
        return k

    @classmethod
    def setUpClass(cls):
        kde_utils.setupClass_norm(cls)

    def loo(self, k, name):
        k.bandwidth = bandwidths.crossvalidation()
        est = k.fit()
        assert est.bandwidth > 0

    def folds(self, k, name):
        imse_args = dict(use_grid=True, folding=5)
        k.bandwidth = bandwidths.crossvalidation(**imse_args)
        est = k.fit()
        assert est.bandwidth > 0

    def imse(self, k, name):
        imse_args = dict(use_grid=True, folding=5)
        k.bandwidth = bandwidths.crossvalidation(bandwidths.CV_IMSE, **imse_args)
        est = k.fit()
        assert est.bandwidth > 0

    def sampling(self, k, name):
        imse_args = dict(sampling=100)
        k.bandwidth = bandwidths.crossvalidation(**imse_args)
        est = k.fit()
        assert est.bandwidth > 0

    def test_loo(self):
        for m in self.methods:
            k = self.createKDE(self.vs[0], m)
            yield self.loo, k, 'loo_{0}_{1}'.format(k.method, 0)

    def test_IMSE(self):
        for m in self.methods:
            k = self.createKDE(self.vs[0], m)
            yield self.imse, k, 'imse_{0}_{1}'.format(k.method, 0)

    def test_folds(self):
        for m in self.methods:
            k = self.createKDE(self.vs[0], m)
            yield self.folds, k, 'folds_{0}_{1}'.format(k.method, 0)

    def test_sampling(self):
        for m in self.methods:
            k = self.createKDE(self.vs[0], m)
            yield self.sampling, k, 'sampling_{0}_{1}'.format(k.method, 0)
