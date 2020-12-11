# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:17:58 2020

Author: Josef Perktold
License: BSD-3

"""

import numpy as np

from statsmodels.tools.decorators import cache_readonly

from statsmodels.stats._diagnostic_other import (
        dispersion_poisson, dispersion_poisson_generic)
from statsmodels.stats.diagnostic_gen import (
        test_chisquare_binning)
from statsmodels.discrete._diagnostics_count import (
        test_poisson_zeroinflation, test_poisson_zeroinflation_brock,
        test_chisquare_prob, plot_probs)


class CountDiagnostic(object):
    pass


class PoissonDiagnostic(CountDiagnostic):
    """Diagnostic and specification tests and plots for Poisson model

    status: very experimental

    Parameters
    ----------
    results : PoissonResults instance

    """

    def __init__(self, results):
        self.results = results

    @cache_readonly
    def probs_predicted(self):
        return self.results.predict(which="prob")

    def test_dispersion(self, ):
        res = dispersion_poisson(self.results)
        return res

    def test_poisson_zeroinflation(self, exog_infl=None):
        res = test_poisson_zeroinflation(self.results, exog_infl=exog_infl)
        return res

    def test_poisson_zeroinflation_brock(self):
        res = test_poisson_zeroinflation_brock(self.results)
        return res

    def test_chisquare_prob(self, bin_edges=None, method=None):
        """moment test for binned probabilites using OPG
        """
        probs = self.results.predict(which="prob")
        res = test_chisquare_prob(self.results, probs, bin_edges=bin_edges,
                                  method=method)
        return res

    def chisquare_binned(self, sort_var=None, bins=10, df=None, ordered=False,
                         sort_method="quicksort", alpha_nc=0.05):
        """Hosmer-Lemeshow style test for count data

        """

        if sort_var is None:
            sort_var = self.results.predict(which="lin")

        endog = self.model.endog
        # not sure yet how this is supposed to work
        # max_count = endog.max * 2
        # no option for max count in predict
        # counts = (endog == np.arange(max_count)).astype(int)
        expected = self.results.predict(which="prob")
        counts = (endog == np.arange(expected.shape[1])).astype(int)
        # we should correct for or include truncated upper bin

        # TODO: what's the correct df, same as for multinomial/ordered ?
        res = test_chisquare_binning(counts, expected, sort_var=sort_var,
                                     bins=bins, df=df, ordered=ordered,
                                     sort_method=sort_method,
                                     alpha_nc=alpha_nc)
        return res

    def plot_probs(self, label='predicted', upp_xlim=None,
                   fig=None):
        """plot observed versus predicted frequencies for entire sample
        """
        probs_predicted = self.probs_predicted.sum(0)
        freq = np.bincount(self.model.endog, minlength=len(probs_predicted))
        fig = plot_probs(freq, label='predicted', upp_xlim=None,
                         fig=None)
        return fig
