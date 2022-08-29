# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 16:15:44 2021

Author: Josef Perktod
License: BSD-3
"""

import numpy as np


class DFamily():
    """Base class for Distribution families.

    """

    def get_distribution(self, *dargs, **dkwds):
        args = self._convert_dargs_sp(*dargs, **dkwds)
        return self.distribution(*args)

    def _convert_dargs_sp(self, *dargs, **dkwds):
        """Convert parameters to scipy distribution parameterization

        Parameters
        ----------
        *dargs : TYPE
            Parameters for family
        **dkwds : TYPE
            DESCRIPTION.

        Returns
        -------
        tuple : Parameters for scipy or scipy compatible distribution class.

        """
        # Note this base class does not convert dargs and ignores dkwds
        return dargs

    def loglike_obs(self, endog, *dargs, **dkwds):
        raise NotImplementedError

    def pdf(self, endog, *dargs, **dkwds):
        return np.exp(self.loglike_obs(endog, *dargs, **dkwds))

    def cdf(self, endog, *dargs, **dkwds):
        raise NotImplementedError

    def score_obs(self, endog, *dargs, **dkwds):
        raise NotImplementedError

    def deriv_pdf(self, endog, *dargs, **dkwds):
        raise NotImplementedError

        # TODO  check vectorization of deriv and numdiff
        from statsmodels.tools.numdiff import _approx_fprime_scalar
        z = np.atleast_1d(endog)

        return _approx_fprime_scalar(z, self.pdf, args=dargs, centered=True)
