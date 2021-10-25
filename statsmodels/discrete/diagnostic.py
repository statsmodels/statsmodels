# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:17:58 2020

Author: Josef Perktold
License: BSD-3

"""

import warnings

import numpy as np

from statsmodels.tools.decorators import cache_readonly

from statsmodels.stats._diagnostic_other import (
    dispersion_poisson,
    # dispersion_poisson_generic,
    )

from statsmodels.stats.diagnostic_gen import (
    test_chisquare_binning
    )
from statsmodels.discrete._diagnostics_count import (
    test_poisson_zeroinflation_jh,
    test_poisson_zeroinflation_broek,
    test_poisson_zeros,
    test_chisquare_prob,
    plot_probs
    )


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

    def test_poisson_zeroinflation(self, method="prob", exog_infl=None):
        """Test for excess zeros, zero inflation or deflation.

        Parameters
        ----------
        method : str
            Three methods ara available for the test:

             = "prob" ; moment test for the probability of zeros
             - "broek" : score test against zero inflation with or without
                explanatory variables for inflation

        exog_infl : array_like or None
            Optional explanatory variables under the alternative of zero
            inflation, or deflation. Only used if method is "broek".

        Returns
        -------
        results

        Notes
        -----
        If method = "prob", then the moment test of He et al 1_ is used based
        on the explicit formula in Tang and Tang 2_.

        If method = "broek" and exog_infl is None, then the test by Van den
        Broek 3_ is used. This is a score test against and alternative of
        constant zero inflation or deflation.

        If method = "broek" and exog_infl is provided, then the extension of
        the broek test to varying zero inflation or deflatio by Jansakul and
        Hinde is used.

        Warning: The Broek test and the Jansakul and Hinde version are not
        stabke when the probability of zeros in Poiss is small, i.e. if the
        conditional means of the estimated Poisson distribution are large.

        """
        if method == "prob":
            if exog_infl is not None:
                warnings.warn('exog_infl is only used if method = "broek"')
            res = test_poisson_zeros(self.results)
        elif method == "broek":
            if exog_infl is None:
                res = test_poisson_zeroinflation_broek(self.results)
            else:
                exog_infl = np.asarray(exog_infl)
                if exog_infl.ndim == 1:
                    exog_infl = exog_infl[:, None]
                res = test_poisson_zeroinflation_jh(self.results,
                                                    exog_infl=exog_infl)

        return res

    def test_chisquare_prob(self, bin_edges=None, method=None):
        """Moment test for binned probabilites using OPG.

        Paramters
        ---------
        binedges : array_like or None
            This defines which counts are included in the test on frequencies
            and how counts are combined in bins.
            The default if bin_edges is None will change in future.
            See Notes and Example sections below.
        method : str
            Currently only `method = "opg"` is available.
            If method is None, the OPG will be used, but the default might
            change in future versions.
            See Notes section below.

        Notes
        -----
        Warning: The current default can have many empty or nearly empty bins.
        The default number of bins is given by max(endog).
        Currently it is recommended to limit the nuber of bins explicitly, s
        ee Examples below.
        Binning will change in future and automatic binning will be added.

        Currently only the outer product of gradient, OPG, method is
        implemented. In many case, the OPG version of a specification test
        overrejects in small samples.
        Specialized tests that use observed or expected information matrix
        often have better small sample properties.
        The default method will change if better methods are added.

        Examples
        --------
        The following call is a test for the probability of zeros
        `test_chisquare_prob(bin_edges=np.arange(3))`

        `test_chisquare_prob(bin_edges=np.arange(10))` tests the hypothesis
        that the frequences for counts up to 7 correspond to the estimated
        Poisson distributions.
        In this case, edges are 0, ..., 9 which defines 9 bins for
        counts 0 to 8. The last bin is dropped, so the joint test hypothesis is
        that the observed aggregated frequencies for counts 0 to 7 correspond
        to the Poisson prediction for those grequencies. Predicted probabilites
        Prob(y_i = k | x) are aggregated over observations ``i``.


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

        endog = self.results.model.endog
        # not sure yet how this is supposed to work
        # max_count = endog.max * 2
        # no option for max count in predict
        # counts = (endog == np.arange(max_count)).astype(int)
        expected = self.results.predict(which="prob")
        expected[:, -1] += 1 - expected.sum(1)
        counts = (endog[:, None] == np.arange(expected.shape[1])).astype(int)
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
        freq = np.bincount(self.results.model.endog.astype(int),
                           minlength=len(probs_predicted))
        fig = plot_probs(freq, probs_predicted,
                         label='predicted', upp_xlim=None,
                         fig=None)
        return fig
