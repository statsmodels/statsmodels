


import numpy as np
from statsmodels.sandbox.survival2 import Survival, KaplanMeier, CoxPH

from numpy.testing import assert_equal


from statsmodels.datasets import ovarian_cancer

dta = ovarian_cancer.load()
darray = np.asarray(dta['data'])


class CheckCoxPH(object):

    def test_smoke_model(self):
        #smoke test to check status and for refactoring
        model = self.model
        results = self.results

        #Note: not available (exception): '_data', 'endog_names', 'exog_names'

        results.baseline()
        results.baseline_object()
        results.conf_int()
        results.cov_params()
        #results.deviance_plot()
        results.diagnostics()
        results.f_test(np.ones(results.params.shape))
        #results.initialize()   #internal
        results.likelihood_ratio_test()
        #results.load()     #inherited pickle load
        results.martingale_plot(1)
        #results.plot()  #BUG: possibly
        results.plot_baseline()
        #results.predict()   #check arguments
        #results.remove_data()
        #results.save()    #inherited pickle save
        results.scheonfeld_plot()
        results.score_test()
        results.summary()
        results.t()
        results.t_test(np.eye(results.params.shape[0]))
        results.test_coefficients()
        results.wald_test()

        assert_equal(results.params.shape, (results.model.exog.shape[1],))
        results.normalized_cov_params
        results.scale
        results.phat
        results.exog_mean
        results.test
        results.test2
        results.params
        results.names
        results.deviance_resid
        results.martingale_resid
        results._cache
        results.model
        results._data_attr
        results.schoenfeld_resid

        results.wald_test()

        #model.initialize() #internal
        #model.fit()  #already called
        model._hessian_proc(results.params)
        model._loglike_proc(results.params)
        model._score_proc(results.params)
        #model._stratify_func() #arguments ?  internal
        model.confint_dist()
        model.covariance(results.params)

        model.hessian(results.params)
        model.information(results.params)

        model.loglike(results.params)
        model.score(results.params)

        #model.predict()
        #results.predict(model.exog[-2:], model.times[-2:])  #BUG
        assert_equal(results.predict('all', 'all').shape, results.model.times.shape)

        model.stratify(1)

        model._str_censoring
        model.df_resid
        model._str_times
        model.confint_dist
        model.d
        model._str_exog
        model._str_d
        model.strata
        model.exog_mean
        model.times
        model.surv
        if self.has_strata:
            model.strata_groups   #not in this example
        model.names
        model.ttype
        model.start_params
        model.ties
        model.censoring
        model.exog

class TestCoxPH1(CheckCoxPH):

    @classmethod
    def setup_class(cls):
        exog = darray[:,range(2,6)]
        surv = Survival(0, censoring=1, data=darray)
        cls.model = model = CoxPH(surv, exog)
        cls.results = model.fit()
        cls.has_strata = False

class TestCoxPHStrata(CheckCoxPH):

    @classmethod
    def setup_class(cls):
        exog = darray[:,range(2,6)]
        surv = Survival(0, censoring=1, data=darray)

        gene1 = exog[:,0]
        expressed = gene1 > gene1.mean()
        ##replacing the column for th first gene with the indicatore variable
        exog[:,0] = expressed
        cls.model = model = CoxPH(surv, exog)
        model.stratify(0,copy=False)
        cls.has_strata = True
        cls.results = model.fit()

if __name__ == '__main__':
    import nose
    tt = TestCoxPH1()
    tt.setup_class()
    tt.test_smoke_model()
    nose.runmodule(argv=[__file__,'-s'], exit=False)
