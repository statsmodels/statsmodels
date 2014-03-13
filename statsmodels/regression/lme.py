"""
Linear mixed effects models for Statsmodels

The data are partitioned into non-overlapping groups, and the model
for group i is:

Y = X*beta + Z*gamma + epsilon

where

* Y is a n_i dimensional response vector
* X is a n_i x p dimensional matrix of fixed effects
  coefficients
* beta is a p-dimensional vector of fixed effects slopes
* Z is a n_i x pr dimensional matrix of random effects
  coefficients
* gamma is a pr-dimensional random vector with mean 0
  and covariance matrix Psi; note that each group
  gets its own independent realization of gamma.
* epsilon is a n_i dimensional vector of iid normal
  errors with mean 0 and variance sigma^2; the epsilon
  values are independent both within and between groups

Y, X and Z must be entirely observed.  beta, Psi, and sigma^2 are
estimated using ML or REML estimation, and gamma and epsilon are
random so define the probability model.

The mean structure is E[Y|X,Z] = X*beta.  If only the mean structure
is of interest, GEE is a good alternative to mixed models.

The primary reference for the implementation details is:

MJ Lindstrom, DM Bates (1988).  "Newton Raphson and EM algorithms for
linear mixed effects models for repeated measures data".  Journal of
the American Statistical Association. Volume 83, Issue 404, pages
1014-1022.

All the likelihood, gradient, and Hessian calculations closely follow
Lindstrom and Bates.

The following two documents are written more from the perspective of
users:

http://lme4.r-forge.r-project.org/lMMwR/lrgprt.pdf

http://lme4.r-forge.r-project.org/slides/2009-07-07-Rennes/3Longitudinal-4.pdf

Notation:

* `revar` is the random effects covariance matrix (Psi above) and
  `sig2` is the (scalar) error variance.  For a single group, the
  marginal covariance matrix of endog given exog is sig2*I + Z * revar
  * Z', where Z is the design matrix for the random effects.

Notes:

1. Three different parameterizations are used here in different
places.  The regression slopes (usually called `params_fe`) are
identical in all three parameterizations, but the variance parameters
differ.  The parameterizations are:

* The "natural parameterization" in which cov(endog) = sig2*I + Z *
  revar * Z', as described above.  This is the main parameterization
  visible to the user.

* The "profile parameterization" in which cov(endog) = I +
  Z * revar1 * Z'.  This is the parameterization of the profile
  likelihood that is maximized to produce parameter estimates.
  (see Lindstrom and Bates for details).  The "natural" revar is
  equal to the "profile" revar1 times sig2.

* The "square root parameterization" in which we work with the
  Cholesky factor of revar1 instead of revar1 directly.

All three parameterizations can be "packed" by concatenating params_fe
together with the lower triangle of the dependence structure.  Note
that when unpacking, it is important to either square or reflect the
dependence structure depending on which parameterization is being
used.

2. The situation where the random effects covariance matrix is
singular is numerically challenging.  Small changes in the covariance
parameters may lead to large changes in the likelihood and
derivatives.

3. The optimization strategy is to optionally perform a few EM steps,
followed by optionally performing a few steepest descent steps,
followed by conjugate gradient descent using one of the scipy gradient
optimizers.  The EM and steepest descent steps are used to get
adequate starting values for the conjugate gradient optimization,
which is much faster.
"""

import numpy as np
import statsmodels.base.model as base
from scipy.optimize import fmin_ncg, fmin_cg, fmin_bfgs, fmin
from scipy.stats.distributions import norm
import pandas as pd
import patsy
#import collections  # OrderedDict requires python >= 2.7
from statsmodels.compatnp.collections import OrderedDict
import warnings
from statsmodels.tools.sm_exceptions import \
     ConvergenceWarning

# Global option to use direct linear algebra calculations for solving
# factor-structured linear systems and calculating factor-structured
# determinants.  Should be False except when testing.
no_smw = False


def smw_solve(f, A, B, BI, rhs):
    """
    Solves the system (f*I + A*B*A') * x = rhs for x and returns x.
    """

    # Direct calculation
    if no_smw:
        mat = np.dot(A, np.dot(B, A.T))
        mat += f*np.eye(A.shape[0])
        return np.linalg.solve(mat, rhs)

    # Use SMW identity
    qmat = BI + np.dot(A.T, A)/f
    u = np.dot(A.T, rhs)
    qmat = np.linalg.solve(qmat, u)
    qmat = np.dot(A, qmat)
    rslt = rhs / f - qmat / f**2
    return rslt


def smw_logdet(f, A, B, BI):
    """
    Use the matrix determinant lemma to accelerate the calculation of
    the log determinant of f*I + A*B*A'.
    """

    if no_smw:
        mat = np.dot(A, np.dot(B, A.T))
        mat += f*np.eye(A.shape[0])
        _, ld = np.linalg.slogdet(mat)
        return ld

    _, ld = np.linalg.slogdet(B)

    p = A.shape[0]
    ld += p*np.log(f)

    qmat = BI + np.dot(A.T, A) / f
    _, ld1 = np.linalg.slogdet(qmat)

    return ld + ld1


class LME(base.Model):

    def __init__(self, endog, exog, groups, exog_re=None,
                 missing='none'):
        """
        Creates an LME object specifying a linear mixed effects model.
        Use the `fit` method to fit the model.

        Arguments:
        ----------
        endog : 1d array-like
            The dependent variable
        exog : 2d array-like
            A matrix of independent variables used to determine the
            mean structure
        groups : 1d array-like
            A vector of labels determining the groups -- data from
            different groups are independent
        exog_re : 2d array-like
            A matrix of independent variables used to determine the
            variance structure.  If None, defaults to a random
            intercept for each of the groups.  May be set from
            a formula using a call to `set_random`.
        missing : string
            The approach to missing data handling
        """

        groups = np.asarray(groups)

        # If there is one covariate, it may be passed in as a column
        # vector, convert these to 2d arrays.
        if exog.ndim == 1:
            exog = exog[:,None]
        if exog_re is not None and exog_re.ndim == 1:
            exog_re = exog_re[:,None]

        self.exog_re_names = None
        if exog_re is None:
            # Default random effects structure (random intercepts).
            exog_re = np.ones((len(endog), 1), dtype=np.float64)
        else:
            try:
                self.exog_re_names = exog_re.columns
            except:
                pass
            self.exog_re_orig = exog_re
            self.exog_re = np.asarray(exog_re)

        # Calling super creates self.endog, etc. as ndarrays and the
        # original exog, endog, etc. are self.data.endog, etc.
        super(LME, self).__init__(endog, exog, groups=groups,
                                  exog_re=exog_re, missing=missing)

        # Model dimensions
        self.p = exog.shape[1]
        if exog_re is not None:
            self.pr = exog_re.shape[1]
            self.prr = int(self.pr * (self.pr + 1) / 2)

        # Convert the data to the internal representation, which is a
        # list of arrays, corresponding to the groups.
        group_labels = list(set(groups))
        group_labels.sort()
        row_indices = dict((s, []) for s in group_labels)
        [row_indices[groups[i]].append(i) for i in range(len(self.endog))]
        self.row_indices = row_indices
        self.group_labels = group_labels
        self.ngroup = len(self.group_labels)

        # Split the data by groups
        self.endog_li = self.group_list(self.endog)
        self.exog_li = self.group_list(self.exog)
        self.exog_re_li = self.group_list(self.exog_re)

        # The total number of observations, summed over all groups
        self.ntot = sum([len(y) for y in self.endog_li])


    def set_random(self, re_formula, data):
        """
        Set the random effects structure using a formula.  This is an
        alternative to providing `exog_re` in the LME constructor.

        Arguments:
        ----------
        re_formula : string
            A string defining the variance structure of the model
            as a formula.
        data : array-like
            The data referenced in re_formula.  Currently must be a
            Pandas DataFrame.

        Notes
        -----
        If the random effects structure is not set either by providing
        `exog_re` to the LME constructor, or by calling `set_random`,
        then the default is a structure with random intercepts for the
        groups.
        """

        # TODO: need a way to process this for missing data
        self.exog_re = patsy.dmatrix(re_formula, data)
        self.exog_re_names = self.exog_re.design_info.column_names
        self.exog_re = np.asarray(self.exog_re)
        self.exog_re_li = self.group_list(self.exog_re)
        self.pr = exog_re.shape
        self.prr = int(self.pr * (self.pr + 1) / 2)

    def group_list(self, array):
        """
        Returns `array` split into subarrays corresponding to the
        grouping structure.
        """

        if array.ndim == 1:
            return [np.array(array[self.row_indices[k]])
                    for k in self.group_labels]
        else:
            return [np.array(array[self.row_indices[k], :])
                    for k in self.group_labels]


    def _unpack(self, params, sym=True):
        """
        Takes as input the packed parameter vector and returns a
        vector containing the regression slopes and a matrix defining
        the dependence structure.

        Arguments:
        ----------
        params : array-like
            The packed parameters
        sym : bool
          If true, the variance parameters are returned as a symmetric
          matrix; if False, the variance parameters are returned as a
          lower triangular matrix.

        Returns:
        --------
        params : 1d ndarray
            The fixed effects coefficients
        revar : 2d ndarray
            The random effects covariance matrix
        """

        params_fe = params[0:self.p]
        params_re = params[self.p:]

        # Unpack the covariance matrix of the random effects
        revar = np.zeros((self.pr, self.pr), dtype=np.float64)
        ix = np.tril_indices(self.pr)
        revar[ix] = params_re

        if sym:
            revar = (revar + revar.T) - np.diag(np.diag(revar))

        return params_fe, revar

    def _pack(self, params_fe, revar):
        """
        Packs the model parameters into a single vector.

        Arguments
        ---------
        params_fe : 1d ndarray
            The fixed effects parameters
        revar : 2d ndarray
            The covariance matrix of the random effects

        Returns
        -------
        params : 1d ndarray
            A vector containing all model parameters, only the lower
            triangle of the random effects covariance matrix is
            included.
        """

        ix = np.tril_indices(revar.shape[0])
        return np.concatenate((params_fe, revar[ix]))

    def like(self, params, reml=True, pen=0.):
        """
        Evaluate the profile log-likelihood of the linear mixed
        effects model.  Specifically, this is the profile likelihood
        in which the scale parameter sig2 has been profiled out.

        Arguments
        ---------
        params : 1d ndarray
            The parameter values, packed into a single vector.  See
            below for details.
        reml : bool
            If true, return REML log likelihood, else return ML
            log likelihood
        pen : non-negative float
            Weight for the penalty parameter for non-SPD covariances

        Returns
        -------
        likeval : scalar
            The log-likelihood value at `params`.

        Notes
        -----
        The first p elements of the packed vector are the regression
        slopes, and the remaining q*(q+1)/2 elements are the lower
        triangle of the random effects covariance matrix Psi, packed
        row-wise.  The matrix Psi is used to form the covariance
        matrix V = I + Z * Psi * Z', where Z is the design matrix for
        the random effects structure.  To convert this to the full
        likelihood (not profiled) parameterization, calculate the
        error variance sig2, and divide Psi by sig2.
        """

        params_fe, revar = self._unpack(params)
        revari = np.linalg.inv(revar)

        if pen > 0:
            cy = np.linalg.cholesky(revar)
            likeval = pen * 2 * np.sum(np.log(np.diag(cy)))
        else:
            likeval =0.
        xvx, qf = 0., 0.
        for k in range(self.ngroup):

            exog = self.exog_li[k]
            ex_r = self.exog_re_li[k]

            # The residuals
            expval = np.dot(exog, params_fe)
            resid = self.endog_li[k] - expval

            # Part 1 of the log likelihood (for both ML and REML)
            ld = smw_logdet(1., ex_r, revar, revari)
            likeval -= ld / 2.

            # Part 2 of the log likelihood (for both ML and REML)
            u = smw_solve(1., ex_r, revar, revari, resid)
            qf += np.dot(resid, u)

            # Adjustment for REML
            if reml:
                mat = smw_solve(1., ex_r, revar, revari, exog)
                xvx += np.dot(exog.T, mat)

        if reml:
            likeval -= (self.ntot - self.p) * np.log(qf) / 2.
            _,ld = np.linalg.slogdet(xvx)
            likeval -= ld / 2.
            likeval -= (self.ntot - self.p) * np.log(2 * np.pi) / 2.
            likeval += ((self.ntot - self.p) *
                        np.log(self.ntot - self.p) / 2.)
            likeval -= (self.ntot - self.p) / 2.
        else:
            likeval -= self.ntot * np.log(qf) / 2.
            likeval -= self.ntot * np.log(2 * np.pi) / 2.
            likeval += self.ntot * np.log(self.ntot) / 2.
            likeval -= self.ntot / 2.

        return likeval

    def _gen_dV_dPsi(self, ex_r, max_ix=None):
        """
        A generator that yields the derivative of the covariance
        matrix V (=I + Z*Psi*Z') with respect to the free elements of
        Psi.  Each call to the generator yields the index of Psi with
        respect to which the derivative is taken, and the derivative
        matrix with respect to that element of Psi.  Psi is a
        symmetric matrix, so the free elements are the lower triangle.
        If max_ix is not None, the iterations terminate after max_ix
        values are yielded.
        """

        jj = 0
        for j1 in range(self.pr):
            for j2 in range(j1 + 1):
                if max_ix is not None and jj > max_ix:
                    return
                mat = np.outer(ex_r[:,j1], ex_r[:,j2])
                if j1 != j2:
                    mat += mat.T
                yield jj,mat
                jj += 1


    def score(self, params, reml=True, pen=0.):
        """
        Calculates the score vector for the mixed effects model.
        Specifically, this is the score for the profile likelihood in
        which the scale parameter sig2 has been profiled out.

        Parameters
        ----------
        params : 1d ndarray
            All model parameters in packed form
        reml : bool
            If true, return REML score, else return ML score
        pen : non-negative float
            Weight for the penalty parameter to deter negative
            variances

        Returns
        -------
        scorevec : 1d ndarray
            The score vector, calculated at `params`.
        """

        params_fe, revar = self._unpack(params)
        revari = np.linalg.inv(revar)
        score_fe = 0.

        score_re = np.zeros(self.prr, dtype=np.float64)

        if pen > 0:
            cy = np.linalg.inv(revar)
            cy = 2*cy - np.diag(np.diag(cy))
            i,j = np.tril_indices(self.pr)
            score_re = pen * cy[i,j]

        rvir = 0.
        rvrb = 0.
        xtvix = 0.
        xtax = [0.,] * self.prr
        B = np.zeros(self.prr, dtype=np.float64)
        C = np.zeros(self.prr, dtype=np.float64)
        for k in range(self.ngroup):

            exog = self.exog_li[k]
            ex_r = self.exog_re_li[k]

            # The residuals
            expval = np.dot(exog, params_fe)
            resid = self.endog_li[k] - expval

            if reml:
                viexog = smw_solve(1., ex_r, revar, revari, exog)
                xtvix += np.dot(exog.T, viexog)

            # Contributions to the covariance parameter gradient
            jj = 0
            vex = smw_solve(1., ex_r, revar, revari, ex_r)
            vir = smw_solve(1., ex_r, revar, revari, resid)
            for jj,mat in self._gen_dV_dPsi(ex_r):
                B[jj] = np.trace(smw_solve(1., ex_r, revar, revari,
                                           mat))
                C[jj] -= np.dot(vir, np.dot(mat, vir))
                if reml:
                    xtax[jj] += np.dot(viexog.T, np.dot(mat, viexog))

            # Contribution of log|V| to the covariance parameter
            # gradient.
            score_re -= 0.5 * B

            # Nededed for the fixed effects params gradient
            rvir += np.dot(resid, vir)
            rvrb -= 2 * np.dot(exog.T, vir)

        fac = self.ntot
        if reml:
            fac -= self.exog.shape[1]

        score_fe = -0.5 * fac * rvrb / rvir

        score_re -= 0.5 * fac * C / rvir

        if reml:
            for j in range(self.prr):
                score_re[j] += 0.5 * np.trace(np.linalg.solve(
                    xtvix, xtax[j]))

        return np.concatenate((score_fe, score_re))


    def like_L(self, params, reml=True, pen=0.):
        """
        Returns the log likelihood evaluated at a given point.  The
        random effects covariance is passed as a square root.

        Arguments:
        ----------
        params : array-like
            The model parameters (for the profile likelihood) in
            packed form.  The first p elements are the regression
            slopes, and the remaining elements are the lower triangle
            of a lower triangular matrix L such that Psi = LL'
        reml : bool
            If true, returns the REML log likelihood, else returns
            the standard log likeihood.
        pen : non-negative float
            The penalty parameter of the logarithmic barrrier
            function for the covariance matrix.

        Returns:
        --------
        The value of the log-likelihood or REML criterion.
        """

        params_fe, L = self._unpack(params, sym=False)
        revar = np.dot(L, L.T)

        params_r = self._pack(params_fe, revar)

        likeval = self.like(params_r, reml, pen)
        return likeval



    def score_L(self, params, reml=True, pen=0.):
        """
        Returns the score vector valuated at a given point.  The
        random effects covariance matrix is passed as a square root.

        Arguments:
        ----------
        params : array-like
            The model parameters (for the profile likelihood) in
            packed form.  The first p elements are the regression
            slopes, and the remaining elements are the lower triangle
            of a lower triangular matrix L such that Psi = LL'
        reml : bool
            If true, returns the REML log likelihood, else returns
            the standard log likeihood.
        pen : non-negative float
            The penalty parameter of the logarithmic barrrier
            function for the covariance matrix.

        Returns:
        --------
        The score vector for the log-likelihood or REML criterion.
        """

        params_fe, L = self._unpack(params, sym=False)
        revar = np.dot(L, L.T)

        params_f = self._pack(params_fe, revar)
        svec = self.score(params_f, reml, pen)
        s_fe, s_re = self._unpack(svec, sym=False)

        # Use the chain rule to get d/dL from d/dPsi
        s_l = np.zeros(self.prr, dtype=np.float64)
        jj = 0
        for i in range(self.pr):
            for j in range(i+1):
                s_l[jj] += np.dot(s_re[:,i], L[:,j])
                s_l[jj] += np.dot(s_re[i,:], L[:,j])
                jj += 1

        gr = np.concatenate((s_fe, s_l))

        return gr

    def hessian(self, params, reml=True, pen=0.):
        """
        Calculates the Hessian matrix for the mixed effects model.
        Specifically, this is the Hessian matrix for the profile
        likelihood in which the scale parameter sig2 has been profiled
        out.  The parameters are passed in packed form, with only the
        lower triangle of the covariance (not its square root) passed.

        Parameters
        ----------
        params : 1d ndarray
            All model parameters in packed form
        reml : bool
            If true, return REML Hessian, else return ML Hessian
        pen : non-negative float
            Weight for the penalty parameter to deter negative
            variances

        Returns
        -------
        hess : 2d ndarray
            The Hessian matrix, evaluated at `params`.
        """

        params_fe, revar = self._unpack(params)
        revari = np.linalg.inv(revar)

        # Blocks for the fixed and random effects parameters.
        hess_fe = 0.
        hess_re = np.zeros((self.prr, self.prr), dtype=np.float64)
        hess_fere = np.zeros((self.prr, self.p), dtype=np.float64)

        fac = self.ntot
        if reml:
            fac -= self.exog.shape[1]

        rvir = 0.
        xtvix = 0.
        xtax = [0.,] * self.prr
        B = np.zeros(self.prr, dtype=np.float64)
        D = np.zeros((self.prr, self.prr), dtype=np.float64)
        F = [[0.,]*self.prr for k in range(self.prr)]
        for k in range(self.ngroup):

            exog = self.exog_li[k]
            ex_r = self.exog_re_li[k]

            # The residuals
            expval = np.dot(exog, params_fe)
            resid = self.endog_li[k] - expval

            viexog = smw_solve(1., ex_r, revar, revari, exog)
            xtvix += np.dot(exog.T, viexog)
            vir = smw_solve(1., ex_r, revar, revari, resid)
            rvir += np.dot(resid, vir)

            for jj1,mat1 in self._gen_dV_dPsi(ex_r):

                hess_fere[jj1,:] += np.dot(viexog.T,
                                           np.dot(mat1, vir))
                if reml:
                    xtax[jj1] += np.dot(viexog.T, np.dot(mat1, viexog))

                B[jj1] += np.dot(vir, np.dot(mat1, vir))
                E = smw_solve(1., ex_r, revar, revari, mat1)

                for jj2,mat2 in self._gen_dV_dPsi(ex_r, jj1):
                    Q = np.dot(mat2, E)
                    Q1 = Q + Q.T
                    vt = np.dot(vir, np.dot(Q1, vir))
                    D[jj1, jj2] += vt
                    if jj1 != jj2:
                        D[jj2, jj1] += vt
                    R = smw_solve(1., ex_r, revar, revari, Q)
                    rt = np.trace(R) / 2
                    hess_re[jj1, jj2] += rt
                    if jj1 != jj2:
                        hess_re[jj2, jj1] += rt
                    if reml:
                        F[jj1][jj2] += np.dot(viexog.T,
                                              np.dot(Q, viexog))

        hess_fe -= fac * xtvix / rvir

        hess_re -= 0.5 * fac * (D / rvir - np.outer(B, B) / rvir**2)

        hess_fere = -fac * hess_fere / rvir

        if reml:
            for j1 in range(self.prr):
                Q1 = np.linalg.solve(xtvix, xtax[j1])
                for j2 in range(j1 + 1):
                    Q2 = np.linalg.solve(xtvix, xtax[j2])
                    a = np.trace(np.dot(Q1, Q2))
                    a -= np.trace(np.linalg.solve(xtvix, F[j1][j2]))
                    a *= 0.5
                    hess_re[j1, j2] += a
                    if j1 > j2:
                        hess_re[j2, j1] += a

        # Put the blocks together to get the Hessian.
        hess = np.zeros((self.p+self.prr, self.p+self.prr),
                        dtype=np.float64)
        hess[0:self.p, 0:self.p] = hess_fe
        hess[0:self.p, self.p:] = hess_fere.T
        hess[self.p:, 0:self.p] = hess_fere
        hess[self.p:, self.p:] = hess_re

        return hess

    def Estep(self, params_fe, revar, sig2):
        """
        The E-step of the EM algorithm.  This is for ML (not REML),
        but it seems to be good enough to use for REML starting
        values.

        Parameters
        ----------
        params_fe : 1d ndarray
            The current value of the fixed effect coefficients
        revar : 2d ndarray
            The current value of the covariance matrix of random
            effects
        sig2 : positive scalar
            The current value of the error variance

        Returns
        -------
        m1x : 1d ndarray
            sum_groups X'*Z*E[gamma | Y], where X and Z are the fixed
            and random effects covariates, gamma is the random
            effects, and Y is the observed data
        m1y : scalar
            sum_groups Y'*E[gamma | Y]
        m2 : 2d ndarray
            sum_groups E[gamma * gamma' | Y]
        m2xx : 2d ndarray
            sum_groups Z'*Z * E[gamma * gamma' | Y]
        """

        m1x, m1y, m2, m2xx = 0., 0., 0., 0.
        revari = np.linalg.inv(revar)

        for k in range(self.ngroup):

            # Get the residuals
            expval = np.dot(self.exog_li[k], params_fe)
            resid = self.endog_li[k] - expval

            # Contruct the marginal covariance matrix for this group
            ex_r = self.exog_re_li[k]

            vr1 = smw_solve(sig2, ex_r, revar, revari, resid)
            vr1 = np.dot(ex_r.T, vr1)
            vr1 = np.dot(revar, vr1)

            vr2 = smw_solve(sig2, ex_r, revar, revari,
                            self.exog_re_li[k])
            vr2 = np.dot(vr2, revar)
            vr2 = np.dot(ex_r.T, vr2)
            vr2 = np.dot(revar, vr2)

            rg = np.dot(ex_r, vr1)
            m1x += np.dot(self.exog_li[k].T, rg)
            m1y += np.dot(self.endog_li[k].T, rg)
            egg = revar - vr2 + np.outer(vr1, vr1)
            m2 += egg
            m2xx += np.dot(np.dot(ex_r.T, ex_r), egg)

        return m1x, m1y, m2, m2xx


    def EM(self, params_fe, revar, sig2, num_em=10,
           hist=None):
        """
        Run the EM algorithm from a given starting point.  This is for
        ML (not REML), but it seems to be good enough to use for REML
        starting values.

        Returns
        -------
        params_fe : 1d ndarray
            The final value of the fixed effects coefficients
        revar : 2d ndarray
            The final value of the random effects covariance
            matrix
        sig2 : float
            The final value of the error variance

        Notes
        -----
        This uses the parameterization of the likelihood sig2*I +
        Z'*V*Z, note that this differs from the profile likelihood
        used in the gradient calculations.
        """

        xxtot = 0.
        for x in self.exog_li:
            xxtot += np.dot(x.T, x)

        xytot = 0.
        for x,y in zip(self.exog_li, self.endog_li):
            xytot += np.dot(x.T, y)

        pp = []
        for itr in range(num_em):

            m1x, m1y, m2, m2xx = self.Estep(params_fe, revar, sig2)

            params_fe = np.linalg.solve(xxtot, xytot - m1x)
            revar = m2 / self.ngroup

            sig2 = 0.
            for x,y in zip(self.exog_li, self.endog_li):
                sig2 += np.sum((y - np.dot(x, params_fe))**2)
            sig2 -= 2*m1y
            sig2 += 2*np.dot(params_fe, m1x)
            sig2 += np.trace(m2xx)
            sig2 /= self.ntot

            if hist is not None:
                hist.append(["EM", params_fe, revar, sig2])

        return params_fe, revar, sig2


    def get_sig2(self, params_fe, revar, reml):
        """
        Returns the estimated error variance based on given estimates
        of the slopes and random effects covariance matrix.

        Arguments:
        ----------
        params_fe : array-like
            The regression slope estimates
        revar : 2d array
            Estimate of the random effects covariance matrix (Psi).
        reml : bool
            If true, use the REML esimate, otherwise use the MLE.

        Returns:
        --------
        sig2 : float
            The estimated error variance.
        """

        revari = np.linalg.inv(revar)

        qf = 0.
        for k in range(self.ngroup):

            exog = self.exog_li[k]
            ex_r = self.exog_re_li[k]

            # The residuals
            expval = np.dot(exog, params_fe)
            resid = self.endog_li[k] - expval

            mat = smw_solve(1., ex_r, revar, revari, resid)
            qf += np.dot(resid, mat)

        if reml:
            qf /= (self.ntot - self.p)
        else:
            qf /= self.ntot

        return qf


    def _steepest_descent(self, func, params, score, gtol=1e-4,
                          max_iter=50):
        """
        Uses the steepest descent algorithm to minimize a function.

        Arguments:
        ----------
        func : function
            The real-valued function to minimize.
        params : array-like
            A point in the domain of `func`, used as the starting
            point for the iterative minimization.
        score : function
            A function implementing the score vector (gradient) of
            `func`.
        gtol : non-negative float
            Return if the sup norm of the score vector is less than
            this value.
        max_iter: non-negative integer
            Return once this number of iterations have occured.

        Returns:
        --------
        params_out : array-like
            The final value of the iterations
        success : bool
            True if the final score vector has sup-norm no larger
            than `gtol`.
        """

        fval = func(params)

        for itr in range(max_iter):

            gro = score(params)
            gr = gro / np.max(np.abs(gro))

            sl = 1.
            success = False
            while sl > 1e-20:
                params1 = params - sl * gr
                fval1 = func(params1)

                if fval1 < fval:
                    params = params1
                    fval = fval1
                    success = True
                    break

                sl /= 2

            if not success:
                break

        return params, np.max(np.abs(gro)) < gtol

    def fit(self, start=None, reml=True, num_sd=2,
            num_em=0, do_cg=True, pen=0., gtol=1e-4, use_L=True,
            free=None, full_output=False):
        """
        Fit a linear mixed model to the data.

        Parameters
        ----------
        start: dict
            If provided, this is a dict containing starting values.
            `start["fe"]` contains starting values for the fixed
            effects regression slopes.  `start["revar"]` contains
            the covariance matrix of random effects as found
            in the `revar` component of LMEResults.  If
            `start["revar"]` is provided, then `start["sig2"]` must
            also be provided (this is the error variance).
            Alternatively, the random effects may be specified as
            `start["revar_L_unscaled"]`, which is the packed lower
            triangle of the covariance matrix in the
            profile parameterization (in this case sig2 is not used).
        reml : bool
            If true, fit according to the REML likelihood, else
            fit the standard likelihood using ML.
        num_sd : integer
            The number of steepest descent iterations
        num_em : non-negative integer
            The number of EM steps.  The EM steps always
            preceed steepest descent and conjugate gradient
            optimization.  The EM algorithm implemented here
            is for ML estimation.
        do_cg : bool
            If True, a conjugate gradient algorithm is
            used for optimization (following any steepest
            descent or EM steps).
        pen : non-negative float
            Coefficient of a logarithmic barrier function
            for SPD covariance and non-negative error variance
        gtol : non-negtive float
            Algorithm is considered converged if the sup-norm of
            the gradient is smaller than this value.
        use_L : bool
            If True, optimization is carried out using the lower
            triangle of the square root of the random effects
            covariance matrix, otherwise it is carried out using the
            lower triangle of the random effects covariance matrix.
        free : tuple of ndarrays
            If not `None`, this is a tuple of length 2 containing 2
            0/1 indicator arrays.  The first element of `free`
            corresponds to the regression slopes and the second
            element of `free` corresponds to the random effects
            covariance matrix (if `use_L` is False) or it square root
            (if `use_L` is True).  A 1 in either array indicates that
            the corresponding parameter is estimated, a 0 indicates
            that it is fixed at its starting value.  One use case if
            to set free[1] to the identity matrix to estimate a model
            with independent random effects.
        full_output : bool
            If true, attach iteration history to results

        Returns
        -------
        A LMEResults instance.
        """

        if use_L:
            like = lambda x: -self.like_L(x, reml, pen)
        else:
            like = lambda x: -self.like(x, reml, pen)

        if free is not None:
            pat_slopes = free[0]
            ix = np.tril_indices(self.exog_re.shape[1])
            pat_revar = free[1][ix]
            pat = np.concatenate((pat_slopes, pat_revar))
            if use_L:
                score = lambda x: -pat*self.score_L(x, reml, pen)
            else:
                score = lambda x: -pat*self.score(x, reml, pen)
        else:
            if use_L:
                score = lambda x: -self.score_L(x, reml, pen)
            else:
                score = lambda x: -self.score(x, reml, pen)

        if full_output:
            hist = []
        else:
            hist = None

        # Starting values
        ix = np.tril_indices(self.pr)
        if start is None:
            start = {}
        if "fe" in start:
            params_fe = start["fe"]
        else:
            params_fe = np.zeros(self.exog.shape[1], dtype=np.float64)
        if "revar_L_unscaled" in start:
            if use_L:
                params_re = start["revar_L_unscaled"]
            else:
                vec = start["revar_L_unscaled"]
                mat = np.zeros((self.pr, self.pr), dtype=np.float64)
                mat[ix] = vec
                mat = np.dot(mat, mat.T)
                params_re = mat[ix]
        elif "revar" in start:
            revar_unscaled = start["revar_scaled"] / start["sig2"]
            if use_L:
                revar_L_unscaled = np.linalg.cholesky(revar_unscaled)
                params_re = revar_L_unscaled[ix]
            else:
                params_re = revar_unscaled[ix]
        else:
            params_re = np.eye(self.exog_re.shape[1])[ix]
        params_prof = np.concatenate((params_fe, params_re))

        success = False

        # EM iterations
        if num_em > 0:
            sig2 = 1.
            params_fe, revar, sig2 = self.EM(params_fe, revar, sig2,
                                             num_em, hist)

            # Gradient algorithms use a different parameterization
            # that profiles out sigma^2.
            if use_L:
                params_prof = self._pack(params_fe, revar / sig2)
            else:
                revar_rt = np.linalg.cholesky(revar / sig2)
                params_prof = self._pack(params_fe, revar_rt)
            if np.max(np.abs(score(params_prof))) < gtol:
                success = True

        for cycle in range(10):

            # Steepest descent iterations
            params_prof, success = self._steepest_descent(like,
                                  params_prof, score,
                                  gtol=gtol, max_iter=num_sd)
            if success:
                break

            # Gradient iterations
            try:
                rslt = fmin_bfgs(like, params_prof, score,
                                 full_output=True,
                                 disp=False,
                                 retall=hist is not None)
            # scipy.optimize routines have trouble staying in the
            # feasible region
            except np.linalg.LinAlgError:
                rslt = None
            if rslt is not None:
                if hist is not None:
                    hist.append(["Gradient", rslt[7]])
                params_prof = rslt[0]
                if np.max(np.abs(score(params_prof))) < gtol:
                    success = True
                    break

        # Convert to the final parameterization (i.e. undo the square
        # root transform of the covariance matrix, and the profiling
        # over the error variance).
        params_fe, revar_ltri = self._unpack(params_prof, sym=False)
        if use_L:
            revar_unscaled = np.dot(revar_ltri, revar_ltri.T)
        else:
            revar_unscaled = revar_ltri
        sig2 = self.get_sig2(params_fe, revar_unscaled, reml)
        revar = sig2 * revar_unscaled

        if not success:
            if rslt is None:
                msg = "Gradient optimization failed, try increasing num_em or pen."
            else:
                msg = "Gradient sup norm=%.3f, try increasing num_em or pen." %\
                      np.max(np.abs(rslt[2]))
            warnings.warn(msg, ConvergenceWarning)

        if np.min(np.abs(np.diag(revar))) < 0.01:
            msg = "The MLE may be on the boundary of the parameter space."
            warnings.warn(msg, ConvergenceWarning)

        # Compute the Hessian at the MLE.  Noe that the hessian
        # function expects the random effects covariance matrix (not
        # its square root).
        params_hess = self._pack(params_fe, revar_unscaled)
        hess = self.hessian(params_hess)
        if free is not None:
            ii = np.flatnonzero(pat)
            hess1 = hess[ii,:][:,ii]
            pcov = np.zeros_like(hess)
            pcov[np.ix_(ii,ii)] = np.linalg.inv(-hess1)
        else:
            pcov = np.linalg.inv(-hess)

        # Prepare a results class instance
        results = LMEResults(self, params_prof, pcov)
        results.params_fe = params_fe
        results.revar = revar
        results.sig2 = sig2
        results.revar_unscaled = revar_unscaled
        results.method = "REML" if reml else "ML"
        results.converged = success
        results.hist = hist
        results.reml = reml
        results.pen = pen
        results.likeval = -like(params_prof)
        results.p = self.p
        results.pr = self.pr
        results.prr = self.prr

        return results




class LMEResults(base.LikelihoodModelResults):
    '''
    Class to contain results of fitting a linear mixed effects model.

    LMEResults inherits from statsmodels.LikelihoodModelResults

    Parameters
    ----------
    See statsmodels.LikelihoodModelResults

    Returns
    -------
    **Attributes**

    model : class instance
        Pointer to PHreg model instance that called fit.
    normalized_cov_params : array
        The sampling covariance matrix of the estimates
    params_fe : array
        The fitted fixed-effects coefficients
    params_re : array
        The fitted random-effects covariance matrix
    bse_fe : array
        The standard errors of the fitted fixed effects coefficients
    bse_re : array
        The standard errors of the fitted random effects covariance
        matrix

    See Also
    --------
    statsmodels.LikelihoodModelResults
    '''


    def __init__(self, model, params, cov_params):

        super(LMEResults, self).__init__(model, params,
           normalized_cov_params=cov_params)

    def bse_fe(self):
        """
        Returns the standard errors of the fixed effect regression
        coefficients.
        """
        p = self.model.exog.shape[1]
        return np.sqrt(np.diag(self.cov_params())[0:p])

    def bse_re(self):
        """
        Returns the standard errors of the variance parameters.  Note
        that the sampling distribution of variance parameters is
        strongly skewed unless the sample size is large, so these
        standard errors may not give meaningful confidence intervals
        of p-values if used in the usual way.
        """
        p = self.model.exog.shape[1]
        return np.sqrt(self.sig2 * np.diag(self.cov_params())[p:])


    def ranef(self):
        """
        Returns posterior means of all random effects.

        Returns:
        --------
        ranef_dict : dict
            A dictionary mapping the distinct values of the `group`
            variable to the conditional means of the random effects
            given the data.
        """

        revari = np.linalg.inv(self.revar)

        ranef_dict = {}
        for k in range(self.model.ngroup):

            endog = self.model.endog_li[k]
            exog = self.model.exog_li[k]
            ex_r = self.model.exog_re_li[k]
            label = self.model.group_labels[k]

            # Get the residuals
            expval = np.dot(exog, self.params_fe)
            resid = endog - expval

            vresid = smw_solve(self.sig2, ex_r, self.revar, revari,
                               resid)

            ranef_dict[label] = np.dot(self.revar,
                                       np.dot(ex_r.T, vresid))

        return ranef_dict


    def ranef_cov(self):
        """
        Returns the conditional covariance matrix of the random
        effects for each group.

        Returns:
        --------
        ranef_dict : dict
            A dictionary mapping the distinct values of the `group`
            variable to the conditional covariance matrix of the
            random effects given the data.
        """

        revari = np.linalg.inv(self.revar)

        ranef_dict = {}
        for k in range(self.model.ngroup):

            endog = self.model.endog_li[k]
            exog = self.model.exog_li[k]
            ex_r = self.model.exog_re_li[k]
            label = self.model.group_labels[k]

            mat1 = np.dot(ex_r, self.revar)
            mat2 = smw_solve(self.sig2, ex_r, self.revar, revari,
                             mat1)
            mat2 = np.dot(mat1.T, mat2)

            ranef_dict[label] = self.revar - mat2

        return ranef_dict


    def summary(self, yname=None, xname_fe=None, xname_re=None,
                title=None, alpha=.05):
        """Summarize the Regression Results

        Parameters
        -----------
        yname : string, optional
            Default is `y`
        xname_fe : list of strings, optional
            Fixed effects covariate names
        xname_re : list of strings, optional
            Random effects covariate names
        title : string, optional
            Title for the top table. If not None, then this replaces
            the default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be
            printed or converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary
            results

        """

        from statsmodels.iolib import summary2
        smry = summary2.Summary()

        info = OrderedDict()
        info["Model:"] = "LME"
        if yname is None:
            yname = self.model.endog_names
        info["Dependent Variable:"] = yname
        info["No. Groups:"] = str(self.model.ngroup)
        info["No. Observations:"] = str(self.model.ntot)
        info["Method:"] = self.method
        info["Res. Var.:"] = self.sig2
        info["Likelihood:"] = self.likeval
        info["Converged:"] = "Yes" if self.converged else "No"
        smry.add_dict(info)

        if xname_fe is not None:
            xname_fe = xname_fe
        elif self.model.exog_names is not None:
            xname_fe = self.model.exog_names
        else:
            xname_fe = []

        if xname_re is not None:
            xname_re = xname_re
        elif self.model.exog_re_names is not None:
            xname_re = self.model.exog_re_names
        else:
            xname_re = []

        while len(xname_fe) < p:
            xname_fe.append("FE%d" % (len(xname_fe) + 1))

        while len(xname_re) < pr:
            xname_re.append("RE%d" % (len(xname_re) + 1))

        float_fmt = "%.3f"

        names = xname_fe
        sdf = np.nan * np.ones((p + self.prr, 6), dtype=np.float64)
        sdf[0:p, 0] = self.params_fe
        sdf[0:p, 1] = np.sqrt(np.diag(self.cov_params()[0:p]))
        sdf[0:p, 2] = sdf[0:p, 0] / sdf[0:p, 1]
        sdf[0:p, 3] = 2 * norm.cdf(-np.abs(sdf[0:p, 2]))
        qm = -norm.ppf(alpha / 2)
        sdf[0:p, 4] = sdf[0:p, 0] - qm * sdf[0:p, 1]
        sdf[0:p, 5] = sdf[0:p, 0] + qm * sdf[0:p, 1]
        jj = p
        for i in range(pr):
            for j in range(i + 1):
                if i == j:
                    names.append(xname_re[i])
                else:
                    names.append(xname_re[i] + " x " + xname_re[j])
                sdf[jj, 0] = self.revar[i, j]
                sdf[jj, 1] = np.sqrt(self.sig2) * self.bse[jj]
                jj += 1

        sdf = pd.DataFrame(index=names, data=sdf)
        sdf.columns = ['Coef.', 'Std.Err.', 'z', 'P>|z|',
                          '[' + str(alpha/2), str(1-alpha/2) + ']']
        for col in sdf.columns:
            sdf[col] = [float_fmt % x if np.isfinite(x) else ""
                        for x in sdf[col]]

        smry.add_df(sdf, align='l')

        return smry


    def profile_re(self, re_ix, num_low=5, dist_low=1., num_high=5,
                   dist_high=1.):
        """
        Calculate a series of values along a 1-dimensional profile
        likelihood.

        Arguments:
        ----------
        re_ix : integer
            The index of the variance parameter for which to construct
            a profile likelihood.
        num_low : integer
            The number of points at which to calculate the likelihood
            below the MLE of the parameter of interest.
        dist_low : float
            The distance below the MLE of the parameter of interest to
            begin calculating points on the profile likelihood.
        num_high : integer
            The number of points at which to calculate the likelihood
            abov the MLE of the parameter of interest.
        dist_high : float
            The distance above the MLE of the parameter of interest to
            begin calculating points on the profile likelihood.

        Result
        ------
        A matrix with two columns.  The first column contains the
        values to which the parameter of interest is constrained.  The
        second column contains the corresponding likelihood values.
        """

        model = self.model
        p = model.exog.shape[1]
        pr = model.exog_re.shape[1]

        # Need to permute the variables so that the profiled variable
        # is first.
        exog_re_li_save = [x.copy() for x in model.exog_re_li]
        ix = range(pr)
        ix[0] = re_ix
        ix[re_ix] = 0
        for k in range(len(model.exog_re_li)):
           model.exog_re_li[k] = model.exog_re_li[k][:,ix]

        # Permute the covariance structure to match the permuted data.
        ru = self.params[p:]
        ik = np.tril_indices(pr)
        mat = np.zeros((pr ,pr), dtype=np.float64)
        mat[ik] = ru
        mat = np.dot(mat, mat.T)
        mat = mat[ix,:][:,ix]
        ix = np.tril_indices(pr)
        params_re = np.linalg.cholesky(mat)[ix]

        # Define the values to which the parameter of interest will be
        # constrained.
        ru0 = params_re[0]
        left = np.linspace(ru0 - dist_low, ru0, num_low + 1)
        right = np.linspace(ru0, ru0 + dist_high, num_high+1)[1:]
        rvalues = np.concatenate((left, right))

        # Indicators of which parameters are free and fixed.
        free_slopes = np.ones(p, dtype=np.float64)
        free_revar = np.ones((pr, pr), dtype=np.float64)
        free_revar[0] = 0

        start = {"fe": self.params_fe}

        likev = []
        for x in rvalues:
            params_re[0] = x
            start["revar_L_unscaled"] = params_re
            md1 = model.fit(start=start,
                            free=(free_slopes, free_revar),
                            reml=self.reml, pen=self.pen)
            likev.append([md1.revar[0,0], md1.likeval])
        likev = np.asarray(likev)

        model.exog_re = exog_re_li_save

        return likev
