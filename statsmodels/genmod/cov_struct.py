from statsmodels.compat.python import iterkeys, itervalues, zip, range
from statsmodels.stats.correlation_tools import cov_nearest
import numpy as np
import pandas as pd
from scipy import linalg as spl
from collections import defaultdict
from statsmodels.tools.sm_exceptions import (ConvergenceWarning, OutputWarning,
                                             NotImplementedWarning)
import warnings

"""
Some details for the covariance calculations can be found in the Stata
docs:

http://www.stata.com/manuals13/xtxtgee.pdf
"""


class CovStruct(object):
    """
    A base class for correlation and covariance structures of grouped
    data.

    Each implementation of this class takes the residuals from a
    regression model that has been fitted to grouped data, and uses
    them to estimate the within-group dependence structure of the
    random errors in the model.

    The state of the covariance structure is represented through the
    value of the class variable `dep_params`.  The default state of a
    newly-created instance should correspond to the identity
    correlation matrix.
    """

    def __init__(self, cov_nearest_method="clipped"):

        # Parameters describing the dependency structure
        self.dep_params = None

        # Keep track of the number of times that the covariance was
        # adjusted.
        self.cov_adjust = []

        # Method for projecting the covariance matrix if it not SPD.
        self.cov_nearest_method = cov_nearest_method

    def initialize(self, model):
        """
        Called by GEE, used by implementations that need additional
        setup prior to running `fit`.

        Parameters
        ----------
        model : GEE class
            A reference to the parent GEE class instance.
        """
        self.model = model

    def update(self, params):
        """
        Updates the association parameter values based on the current
        regression coefficients.

        Parameters
        ----------
        params : array-like
            Working values for the regression parameters.
        """
        raise NotImplementedError

    def covariance_matrix(self, endog_expval, index):
        """
        Returns the working covariance or correlation matrix for a
        given cluster of data.

        Parameters
        ----------
        endog_expval: array-like
           The expected values of endog for the cluster for which the
           covariance or correlation matrix will be returned
        index: integer
           The index of the cluster for which the covariane or
           correlation matrix will be returned

        Returns
        -------
        M: matrix
            The covariance or correlation matrix of endog
        is_cor: bool
            True if M is a correlation matrix, False if M is a
            covariance matrix
        """
        raise NotImplementedError

    def covariance_matrix_solve(self, expval, index, stdev, rhs):
        """
        Solves matrix equations of the form `covmat * soln = rhs` and
        returns the values of `soln`, where `covmat` is the covariance
        matrix represented by this class.

        Parameters
        ----------
        expval: array-like
           The expected value of endog for each observed value in the
           group.
        index: integer
           The group index.
        stdev : array-like
            The standard deviation of endog for each observation in
            the group.
        rhs : list/tuple of array-like
            A set of right-hand sides; each defines a matrix equation
            to be solved.

        Returns
        -------
        soln : list/tuple of array-like
            The solutions to the matrix equations.

        Notes
        -----
        Returns None if the solver fails.

        Some dependence structures do not use `expval` and/or `index`
        to determine the correlation matrix.  Some families
        (e.g. binomial) do not use the `stdev` parameter when forming
        the covariance matrix.

        If the covariance matrix is singular or not SPD, it is
        projected to the nearest such matrix.  These projection events
        are recorded in the fit_history member of the GEE model.

        Systems of linear equations with the covariance matrix as the
        left hand side (LHS) are solved for different right hand sides
        (RHS); the LHS is only factorized once to save time.

        This is a default implementation, it can be reimplemented in
        subclasses to optimize the linear algebra according to the
        struture of the covariance matrix.
        """

        vmat, is_cor = self.covariance_matrix(expval, index)
        if is_cor:
            vmat *= np.outer(stdev, stdev)

        # Factor the covariance matrix.  If the factorization fails,
        # attempt to condition it into a factorizable matrix.
        threshold = 1e-2
        success = False
        cov_adjust = 0
        for itr in range(20):
            try:
                vco = spl.cho_factor(vmat)
                success = True
                break
            except np.linalg.LinAlgError:
                vmat = cov_nearest(vmat, method=self.cov_nearest_method,
                                   threshold=threshold)
                threshold *= 2
                cov_adjust += 1

        self.cov_adjust.append(cov_adjust)

        # Last resort if we still can't factor the covariance matrix.
        if not success:
            warnings.warn(
                "Unable to condition covariance matrix to an SPD "
                "matrix using cov_nearest", ConvergenceWarning)
            vmat = np.diag(np.diag(vmat))
            vco = spl.cho_factor(vmat)

        soln = [spl.cho_solve(vco, x) for x in rhs]
        return soln

    def summary(self):
        """
        Returns a text summary of the current estimate of the
        dependence structure.
        """
        raise NotImplementedError


class Independence(CovStruct):
    """
    An independence working dependence structure.
    """

    # Nothing to update
    def update(self, params):
        return

    def covariance_matrix(self, expval, index):
        dim = len(expval)
        return np.eye(dim, dtype=np.float64), True

    def covariance_matrix_solve(self, expval, index, stdev, rhs):
        v = stdev ** 2
        rslt = []
        for x in rhs:
            if x.ndim == 1:
                rslt.append(x / v)
            else:
                rslt.append(x / v[:, None])
        return rslt

    update.__doc__ = CovStruct.update.__doc__
    covariance_matrix.__doc__ = CovStruct.covariance_matrix.__doc__
    covariance_matrix_solve.__doc__ = CovStruct.covariance_matrix_solve.__doc__

    def summary(self):
        return ("Observations within a cluster are modeled "
                "as being independent.")


class Exchangeable(CovStruct):
    """
    An exchangeable working dependence structure.
    """

    def __init__(self):

        super(Exchangeable, self).__init__()

        # The correlation between any two values in the same cluster
        self.dep_params = 0.

    def update(self, params):

        endog = self.model.endog_li

        nobs = self.model.nobs

        varfunc = self.model.family.variance

        cached_means = self.model.cached_means

        has_weights = self.model.weights is not None
        weights_li = self.model.weights

        residsq_sum, scale = 0, 0
        fsum1, fsum2, n_pairs = 0., 0., 0.
        for i in range(self.model.num_group):
            expval, _ = cached_means[i]
            stdev = np.sqrt(varfunc(expval))
            resid = (endog[i] - expval) / stdev
            f = weights_li[i] if has_weights else 1.

            ssr = np.sum(resid * resid)
            scale += f * ssr
            fsum1 += f * len(endog[i])

            residsq_sum += f * (resid.sum() ** 2 - ssr) / 2
            ngrp = len(resid)
            npr = 0.5 * ngrp * (ngrp - 1)
            fsum2 += f * npr
            n_pairs += npr

        ddof = self.model.ddof_scale
        scale /= (fsum1 * (nobs - ddof) / float(nobs))
        residsq_sum /= scale
        self.dep_params = residsq_sum / \
            (fsum2 * (n_pairs - ddof) / float(n_pairs))

    def covariance_matrix(self, expval, index):
        dim = len(expval)
        dp = self.dep_params * np.ones((dim, dim), dtype=np.float64)
        np.fill_diagonal(dp, 1)
        return dp, True

    def covariance_matrix_solve(self, expval, index, stdev, rhs):

        k = len(expval)
        c = self.dep_params / (1. - self.dep_params)
        c /= 1. + self.dep_params * (k - 1)

        rslt = []
        for x in rhs:
            if x.ndim == 1:
                x1 = x / stdev
                y = x1 / (1. - self.dep_params)
                y -= c * sum(x1)
                y /= stdev
            else:
                x1 = x / stdev[:, None]
                y = x1 / (1. - self.dep_params)
                y -= c * x1.sum(0)
                y /= stdev[:, None]
            rslt.append(y)

        return rslt

    update.__doc__ = CovStruct.update.__doc__
    covariance_matrix.__doc__ = CovStruct.covariance_matrix.__doc__
    covariance_matrix_solve.__doc__ = CovStruct.covariance_matrix_solve.__doc__

    def summary(self):
        return ("The correlation between two observations in the " +
                "same cluster is %.3f" % self.dep_params)


class Nested(CovStruct):
    """
    A nested working dependence structure.

    A working dependence structure that captures a nested hierarchy of
    groups, each level of which contributes to the random error term
    of the model.

    When using this working covariance structure, `dep_data` of the
    GEE instance should contain a n_obs x k matrix of 0/1 indicators,
    corresponding to the k subgroups nested under the top-level
    `groups` of the GEE instance.  These subgroups should be nested
    from left to right, so that two observations with the same value
    for column j of `dep_data` should also have the same value for all
    columns j' < j (this only applies to observations in the same
    top-level cluster given by the `groups` argument to GEE).

    Examples
    --------
    Suppose our data are student test scores, and the students are in
    classrooms, nested in schools, nested in school districts.  The
    school district is the highest level of grouping, so the school
    district id would be provided to GEE as `groups`, and the school
    and classroom id's would be provided to the Nested class as the
    `dep_data` argument, e.g.

        0 0  # School 0, classroom 0, student 0
        0 0  # School 0, classroom 0, student 1
        0 1  # School 0, classroom 1, student 0
        0 1  # School 0, classroom 1, student 1
        1 0  # School 1, classroom 0, student 0
        1 0  # School 1, classroom 0, student 1
        1 1  # School 1, classroom 1, student 0
        1 1  # School 1, classroom 1, student 1

    Labels lower in the hierarchy are recycled, so that student 0 in
    classroom 0 is different fro student 0 in classroom 1, etc.

    Notes
    -----
    The calculations for this dependence structure involve all pairs
    of observations within a group (that is, within the top level
    `group` structure passed to GEE).  Large group sizes will result
    in slow iterations.

    The variance components are estimated using least squares
    regression of the products r*r', for standardized residuals r and
    r' in the same group, on a vector of indicators defining which
    variance components are shared by r and r'.
    """

    def initialize(self, model):
        """
        Called on the first call to update

        `ilabels` is a list of n_i x n_i matrices containing integer
        labels that correspond to specific correlation parameters.
        Two elements of ilabels[i] with the same label share identical
        variance components.

        `designx` is a matrix, with each row containing dummy
        variables indicating which variance components are associated
        with the corresponding element of QY.
        """

        super(Nested, self).initialize(model)

        if self.model.weights is not None:
            warnings.warn("weights not implemented for nested cov_struct, "
                          "using unweighted covariance estimate",
                          NotImplementedWarning)

        # A bit of processing of the nest data
        id_matrix = np.asarray(self.model.dep_data)
        if id_matrix.ndim == 1:
            id_matrix = id_matrix[:, None]
        self.id_matrix = id_matrix

        endog = self.model.endog_li
        designx, ilabels = [], []

        # The number of layers of nesting
        n_nest = self.id_matrix.shape[1]

        for i in range(self.model.num_group):
            ngrp = len(endog[i])
            glab = self.model.group_labels[i]
            rix = self.model.group_indices[glab]

            # Determine the number of common variance components
            # shared by each pair of observations.
            ix1, ix2 = np.tril_indices(ngrp, -1)
            ncm = (self.id_matrix[rix[ix1], :] ==
                   self.id_matrix[rix[ix2], :]).sum(1)

            # This is used to construct the working correlation
            # matrix.
            ilabel = np.zeros((ngrp, ngrp), dtype=np.int32)
            ilabel[[ix1, ix2]] = ncm + 1
            ilabel[[ix2, ix1]] = ncm + 1
            ilabels.append(ilabel)

            # This is used to estimate the variance components.
            dsx = np.zeros((len(ix1), n_nest + 1), dtype=np.float64)
            dsx[:, 0] = 1
            for k in np.unique(ncm):
                ii = np.flatnonzero(ncm == k)
                dsx[ii, 1:k + 1] = 1
            designx.append(dsx)

        self.designx = np.concatenate(designx, axis=0)
        self.ilabels = ilabels

        svd = np.linalg.svd(self.designx, 0)
        self.designx_u = svd[0]
        self.designx_s = svd[1]
        self.designx_v = svd[2].T

    def update(self, params):

        endog = self.model.endog_li

        nobs = self.model.nobs
        dim = len(params)

        if self.designx is None:
            self._compute_design(self.model)

        cached_means = self.model.cached_means

        varfunc = self.model.family.variance

        dvmat = []
        scale = 0.
        for i in range(self.model.num_group):

            expval, _ = cached_means[i]

            stdev = np.sqrt(varfunc(expval))
            resid = (endog[i] - expval) / stdev

            ix1, ix2 = np.tril_indices(len(resid), -1)
            dvmat.append(resid[ix1] * resid[ix2])

            scale += np.sum(resid ** 2)

        dvmat = np.concatenate(dvmat)
        scale /= (nobs - dim)

        # Use least squares regression to estimate the variance
        # components
        vcomp_coeff = np.dot(self.designx_v, np.dot(self.designx_u.T,
                                                    dvmat) / self.designx_s)

        self.vcomp_coeff = np.clip(vcomp_coeff, 0, np.inf)
        self.scale = scale

        self.dep_params = self.vcomp_coeff.copy()

    def covariance_matrix(self, expval, index):

        dim = len(expval)

        # First iteration
        if self.dep_params is None:
            return np.eye(dim, dtype=np.float64), True

        ilabel = self.ilabels[index]

        c = np.r_[self.scale, np.cumsum(self.vcomp_coeff)]
        vmat = c[ilabel]
        vmat /= self.scale
        return vmat, True

    update.__doc__ = CovStruct.update.__doc__
    covariance_matrix.__doc__ = CovStruct.covariance_matrix.__doc__

    def summary(self):
        """
        Returns a summary string describing the state of the
        dependence structure.
        """

        msg = "Variance estimates\n------------------\n"
        for k in range(len(self.vcomp_coeff)):
            msg += "Component %d: %.3f\n" % (k + 1, self.vcomp_coeff[k])
        msg += "Residual: %.3f\n" % (self.scale -
                                     np.sum(self.vcomp_coeff))
        return msg


class Stationary(CovStruct):
    """
    A stationary covariance structure.

    The correlation between two observations is an arbitrary function
    of the distance between them.  Distances up to a given maximum
    value are included in the covariance model.

    Parameters
    ----------
    max_lag : float
        The largest distance that is included in the covariance model.
    grid : bool
        If True, the index positions in the data (after dropping missing
        values) are used to define distances, and the `time` variable is
        ignored.
    """

    def __init__(self, max_lag=1, grid=False):

        super(Stationary, self).__init__()
        self.max_lag = max_lag
        self.grid = grid
        self.dep_params = np.zeros(max_lag)

    def initialize(self, model):

        super(Stationary, self).initialize(model)

        # Time used as an index needs to be integer type.
        if not self.grid:
            time = self.model.time[:, 0].astype(np.int32)
            self.time = self.model.cluster_list(time)

    def update(self, params):

        if self.grid:
            self.update_grid(params)
        else:
            self.update_nogrid(params)

    def update_grid(self, params):

        endog = self.model.endog_li
        cached_means = self.model.cached_means
        varfunc = self.model.family.variance

        dep_params = np.zeros(self.max_lag + 1)
        for i in range(self.model.num_group):

            expval, _ = cached_means[i]
            stdev = np.sqrt(varfunc(expval))
            resid = (endog[i] - expval) / stdev

            dep_params[0] += np.sum(resid * resid) / len(resid)
            for j in range(1, self.max_lag + 1):
                dep_params[j] += np.sum(resid[0:-j] *
                                        resid[j:]) / len(resid[j:])

        self.dep_params = dep_params[1:] / dep_params[0]

    def update_nogrid(self, params):

        endog = self.model.endog_li
        cached_means = self.model.cached_means
        varfunc = self.model.family.variance

        dep_params = np.zeros(self.max_lag + 1)
        dn = np.zeros(self.max_lag + 1)
        for i in range(self.model.num_group):

            expval, _ = cached_means[i]
            stdev = np.sqrt(varfunc(expval))
            resid = (endog[i] - expval) / stdev

            j1, j2 = np.tril_indices(len(expval))
            dx = np.abs(self.time[i][j1] - self.time[i][j2])
            ii = np.flatnonzero(dx <= self.max_lag)
            j1 = j1[ii]
            j2 = j2[ii]
            dx = dx[ii]

            vs = np.bincount(dx, weights=resid[
                             j1] * resid[j2], minlength=self.max_lag + 1)
            vd = np.bincount(dx, minlength=self.max_lag + 1)

            ii = np.flatnonzero(vd > 0)
            dn[ii] += 1
            if len(ii) > 0:
                dep_params[ii] += vs[ii] / vd[ii]

        dep_params /= dn
        self.dep_params = dep_params[1:] / dep_params[0]

    def covariance_matrix(self, endog_expval, index):

        if self.grid:
            return self.covariance_matrix_grid(endog_expval, index)

        j1, j2 = np.tril_indices(len(endog_expval))
        dx = np.abs(self.time[index][j1] - self.time[index][j2])
        ii = np.flatnonzero((0 < dx) & (dx <= self.max_lag))
        j1 = j1[ii]
        j2 = j2[ii]
        dx = dx[ii]

        cmat = np.eye(len(endog_expval))
        cmat[j1, j2] = self.dep_params[dx - 1]
        cmat[j2, j1] = self.dep_params[dx - 1]
        return cmat, True

    def covariance_matrix_grid(self, endog_expval, index):

        from scipy.linalg import toeplitz
        r = np.zeros(len(endog_expval))
        r[0] = 1
        r[1:self.max_lag + 1] = self.dep_params
        return toeplitz(r), True

    def covariance_matrix_solve(self, expval, index, stdev, rhs):

        if not self.grid:
            return super(Stationary, self).covariance_matrix_solve(
                expval, index, stdev, rhs)

        from statsmodels.tools.linalg import stationary_solve
        r = np.zeros(len(expval))
        r[0:self.max_lag] = self.dep_params
        return [stationary_solve(r, x) for x in rhs]

    update.__doc__ = CovStruct.update.__doc__
    covariance_matrix.__doc__ = CovStruct.covariance_matrix.__doc__
    covariance_matrix_solve.__doc__ = CovStruct.covariance_matrix_solve.__doc__

    def summary(self):

        return ("Stationary dependence parameters\n",
                self.dep_params)


class Autoregressive(CovStruct):
    """
    A first-order autoregressive working dependence structure.

    The dependence is defined in terms of the `time` component of the
    parent GEE class, which defaults to the index position of each
    value within its cluster, based on the order of values in the
    input data set.  Time represents a potentially multidimensional
    index from which distances between pairs of observations can be
    determined.

    The correlation between two observations in the same cluster is
    dep_params^distance, where `dep_params` contains the (scalar)
    autocorrelation parameter to be estimated, and `distance` is the
    distance between the two observations, calculated from their
    corresponding time values.  `time` is stored as an n_obs x k
    matrix, where `k` represents the number of dimensions in the time
    index.

    The autocorrelation parameter is estimated using weighted
    nonlinear least squares, regressing each value within a cluster on
    each preceeding value in the same cluster.

    Parameters
    ----------
    dist_func: function from R^k x R^k to R^+, optional
        A function that computes the distance between the two
        observations based on their `time` values.

    References
    ----------
    B Rosner, A Munoz.  Autoregressive modeling for the analysis of
    longitudinal data with unequally spaced examinations.  Statistics
    in medicine. Vol 7, 59-71, 1988.
    """

    def __init__(self, dist_func=None):

        super(Autoregressive, self).__init__()

        # The function for determining distances based on time
        if dist_func is None:
            self.dist_func = lambda x, y: np.abs(x - y).sum()
        else:
            self.dist_func = dist_func

        self.designx = None

        # The autocorrelation parameter
        self.dep_params = 0.

    def update(self, params):

        if self.model.weights is not None:
            warnings.warn("weights not implemented for autoregressive "
                          "cov_struct, using unweighted covariance estimate",
                          NotImplementedWarning)

        endog = self.model.endog_li
        time = self.model.time_li

        # Only need to compute this once
        if self.designx is not None:
            designx = self.designx
        else:
            designx = []
            for i in range(self.model.num_group):

                ngrp = len(endog[i])
                if ngrp == 0:
                    continue

                # Loop over pairs of observations within a cluster
                for j1 in range(ngrp):
                    for j2 in range(j1):
                        designx.append(self.dist_func(time[i][j1, :],
                                                      time[i][j2, :]))

            designx = np.array(designx)
            self.designx = designx

        scale = self.model.estimate_scale()
        varfunc = self.model.family.variance
        cached_means = self.model.cached_means

        # Weights
        var = 1. - self.dep_params ** (2 * designx)
        var /= 1. - self.dep_params ** 2
        wts = 1. / var
        wts /= wts.sum()

        residmat = []
        for i in range(self.model.num_group):

            expval, _ = cached_means[i]
            stdev = np.sqrt(scale * varfunc(expval))
            resid = (endog[i] - expval) / stdev

            ngrp = len(resid)
            for j1 in range(ngrp):
                for j2 in range(j1):
                    residmat.append([resid[j1], resid[j2]])

        residmat = np.array(residmat)

        # Need to minimize this
        def fitfunc(a):
            dif = residmat[:, 0] - (a ** designx) * residmat[:, 1]
            return np.dot(dif ** 2, wts)

        # Left bracket point
        b_lft, f_lft = 0., fitfunc(0.)

        # Center bracket point
        b_ctr, f_ctr = 0.5, fitfunc(0.5)
        while f_ctr > f_lft:
            b_ctr /= 2
            f_ctr = fitfunc(b_ctr)
            if b_ctr < 1e-8:
                self.dep_params = 0
                return

        # Right bracket point
        b_rgt, f_rgt = 0.75, fitfunc(0.75)
        while f_rgt < f_ctr:
            b_rgt = b_rgt + (1. - b_rgt) / 2
            f_rgt = fitfunc(b_rgt)
            if b_rgt > 1. - 1e-6:
                raise ValueError(
                    "Autoregressive: unable to find right bracket")

        from scipy.optimize import brent
        self.dep_params = brent(fitfunc, brack=[b_lft, b_ctr, b_rgt])

    def covariance_matrix(self, endog_expval, index):
        ngrp = len(endog_expval)
        if self.dep_params == 0:
            return np.eye(ngrp, dtype=np.float64), True
        idx = np.arange(ngrp)
        cmat = self.dep_params ** np.abs(idx[:, None] - idx[None, :])
        return cmat, True

    def covariance_matrix_solve(self, expval, index, stdev, rhs):
        # The inverse of an AR(1) covariance matrix is tri-diagonal.

        k = len(expval)
        soln = []

        # LHS has 1 column
        if k == 1:
            return [x / stdev ** 2 for x in rhs]

        # LHS has 2 columns
        if k == 2:
            mat = np.array([[1, -self.dep_params], [-self.dep_params, 1]])
            mat /= (1. - self.dep_params ** 2)
            for x in rhs:
                if x.ndim == 1:
                    x1 = x / stdev
                else:
                    x1 = x / stdev[:, None]
                x1 = np.dot(mat, x1)
                if x.ndim == 1:
                    x1 /= stdev
                else:
                    x1 /= stdev[:, None]
                soln.append(x1)
            return soln

        # LHS has >= 3 columns: values c0, c1, c2 defined below give
        # the inverse.  c0 is on the diagonal, except for the first
        # and last position.  c1 is on the first and last position of
        # the diagonal.  c2 is on the sub/super diagonal.
        c0 = (1. + self.dep_params ** 2) / (1. - self.dep_params ** 2)
        c1 = 1. / (1. - self.dep_params ** 2)
        c2 = -self.dep_params / (1. - self.dep_params ** 2)
        soln = []
        for x in rhs:
            flatten = False
            if x.ndim == 1:
                x = x[:, None]
                flatten = True
            x1 = x / stdev[:, None]

            z0 = np.zeros((1, x.shape[1]))
            rhs1 = np.concatenate((x[1:, :], z0), axis=0)
            rhs2 = np.concatenate((z0, x[0:-1, :]), axis=0)

            y = c0 * x + c2 * rhs1 + c2 * rhs2
            y[0, :] = c1 * x[0, :] + c2 * x[1, :]
            y[-1, :] = c1 * x[-1, :] + c2 * x[-2, :]

            y /= stdev[:, None]

            if flatten:
                y = np.squeeze(y)

            soln.append(y)

        return soln

    update.__doc__ = CovStruct.update.__doc__
    covariance_matrix.__doc__ = CovStruct.covariance_matrix.__doc__
    covariance_matrix_solve.__doc__ = CovStruct.covariance_matrix_solve.__doc__

    def summary(self):

        return ("Autoregressive(1) dependence parameter: %.3f\n" %
                self.dep_params)


class CategoricalCovStruct(CovStruct):
    """
    Parent class for covariance structure for categorical data models.

    Attributes
    ----------
    nlevel : int
        The number of distinct levels for the outcome variable.
    ibd : list
        A list whose i^th element ibd[i] is an array whose rows
        contain integer pairs (a,b), where endog_li[i][a:b] is the
        subvector of binary indicators derived from the same ordinal
        value.
    """

    def initialize(self, model):

        super(CategoricalCovStruct, self).initialize(model)

        self.nlevel = len(model.endog_values)
        self._ncut = self.nlevel - 1

        from numpy.lib.stride_tricks import as_strided
        b = np.dtype(np.int64).itemsize

        ibd = []
        for v in model.endog_li:
            jj = np.arange(0, len(v) + 1, self._ncut, dtype=np.int64)
            jj = as_strided(jj, shape=(len(jj) - 1, 2), strides=(b, b))
            ibd.append(jj)

        self.ibd = ibd


class GlobalOddsRatio(CategoricalCovStruct):
    """
    Estimate the global odds ratio for a GEE with ordinal or nominal
    data.

    References
    ----------
    PJ Heagerty and S Zeger. "Marginal Regression Models for Clustered
    Ordinal Measurements". Journal of the American Statistical
    Association Vol. 91, Issue 435 (1996).

    Thomas Lumley. Generalized Estimating Equations for Ordinal Data:
    A Note on Working Correlation Structures. Biometrics Vol. 52,
    No. 1 (Mar., 1996), pp. 354-361
    http://www.jstor.org/stable/2533173

    Notes
    -----
    The following data structures are calculated in the class:

    'ibd' is a list whose i^th element ibd[i] is a sequence of integer
    pairs (a,b), where endog_li[i][a:b] is the subvector of binary
    indicators derived from the same ordinal value.

    `cpp` is a dictionary where cpp[group] is a map from cut-point
    pairs (c,c') to the indices of all between-subject pairs derived
    from the given cut points.
    """

    def __init__(self, endog_type):
        super(GlobalOddsRatio, self).__init__()
        self.endog_type = endog_type
        self.dep_params = 0.

    def initialize(self, model):

        super(GlobalOddsRatio, self).initialize(model)

        if self.model.weights is not None:
            warnings.warn("weights not implemented for GlobalOddsRatio "
                          "cov_struct, using unweighted covariance estimate",
                          NotImplementedWarning)

        # Need to restrict to between-subject pairs
        cpp = []
        for v in model.endog_li:

            # Number of subjects in this group
            m = int(len(v) / self._ncut)
            i1, i2 = np.tril_indices(m, -1)

            cpp1 = {}
            for k1 in range(self._ncut):
                for k2 in range(k1 + 1):
                    jj = np.zeros((len(i1), 2), dtype=np.int64)
                    jj[:, 0] = i1 * self._ncut + k1
                    jj[:, 1] = i2 * self._ncut + k2
                    cpp1[(k2, k1)] = jj

            cpp.append(cpp1)

        self.cpp = cpp

        # Initialize the dependence parameters
        self.crude_or = self.observed_crude_oddsratio()
        if self.model.update_dep:
            self.dep_params = self.crude_or

    def pooled_odds_ratio(self, tables):
        """
        Returns the pooled odds ratio for a list of 2x2 tables.

        The pooled odds ratio is the inverse variance weighted average
        of the sample odds ratios of the tables.
        """

        if len(tables) == 0:
            return 1.

        # Get the sampled odds ratios and variances
        log_oddsratio, var = [], []
        for table in tables:
            lor = np.log(table[1, 1]) + np.log(table[0, 0]) -\
                np.log(table[0, 1]) - np.log(table[1, 0])
            log_oddsratio.append(lor)
            var.append((1 / table.astype(np.float64)).sum())

        # Calculate the inverse variance weighted average
        wts = [1 / v for v in var]
        wtsum = sum(wts)
        wts = [w / wtsum for w in wts]
        log_pooled_or = sum([w * e for w, e in zip(wts, log_oddsratio)])

        return np.exp(log_pooled_or)

    def covariance_matrix(self, expected_value, index):

        vmat = self.get_eyy(expected_value, index)
        vmat -= np.outer(expected_value, expected_value)
        return vmat, False

    def observed_crude_oddsratio(self):
        """
        To obtain the crude (global) odds ratio, first pool all binary
        indicators corresponding to a given pair of cut points (c,c'),
        then calculate the odds ratio for this 2x2 table.  The crude
        odds ratio is the inverse variance weighted average of these
        odds ratios.  Since the covariate effects are ignored, this OR
        will generally be greater than the stratified OR.
        """

        cpp = self.cpp
        endog = self.model.endog_li

        # Storage for the contingency tables for each (c,c')
        tables = {}
        for ii in iterkeys(cpp[0]):
            tables[ii] = np.zeros((2, 2), dtype=np.float64)

        # Get the observed crude OR
        for i in range(len(endog)):

            # The observed joint values for the current cluster
            yvec = endog[i]
            endog_11 = np.outer(yvec, yvec)
            endog_10 = np.outer(yvec, 1. - yvec)
            endog_01 = np.outer(1. - yvec, yvec)
            endog_00 = np.outer(1. - yvec, 1. - yvec)

            cpp1 = cpp[i]
            for ky in iterkeys(cpp1):
                ix = cpp1[ky]
                tables[ky][1, 1] += endog_11[ix[:, 0], ix[:, 1]].sum()
                tables[ky][1, 0] += endog_10[ix[:, 0], ix[:, 1]].sum()
                tables[ky][0, 1] += endog_01[ix[:, 0], ix[:, 1]].sum()
                tables[ky][0, 0] += endog_00[ix[:, 0], ix[:, 1]].sum()

        return self.pooled_odds_ratio(list(itervalues(tables)))

    def get_eyy(self, endog_expval, index):
        """
        Returns a matrix V such that V[i,j] is the joint probability
        that endog[i] = 1 and endog[j] = 1, based on the marginal
        probabilities of endog and the global odds ratio `current_or`.
        """

        current_or = self.dep_params
        ibd = self.ibd[index]

        # The between-observation joint probabilities
        if current_or == 1.0:
            vmat = np.outer(endog_expval, endog_expval)
        else:
            psum = endog_expval[:, None] + endog_expval[None, :]
            pprod = endog_expval[:, None] * endog_expval[None, :]
            pfac = np.sqrt((1. + psum * (current_or - 1.)) ** 2 +
                           4 * current_or * (1. - current_or) * pprod)
            vmat = 1. + psum * (current_or - 1.) - pfac
            vmat /= 2. * (current_or - 1)

        # Fix E[YY'] for elements that belong to same observation
        for bdl in ibd:
            evy = endog_expval[bdl[0]:bdl[1]]
            if self.endog_type == "ordinal":
                vmat[bdl[0]:bdl[1], bdl[0]:bdl[1]] =\
                    np.minimum.outer(evy, evy)
            else:
                vmat[bdl[0]:bdl[1], bdl[0]:bdl[1]] = np.diag(evy)

        return vmat

    def update(self, params):
        """
        Update the global odds ratio based on the current value of
        params.
        """

        cpp = self.cpp
        cached_means = self.model.cached_means

        # This will happen if all the clusters have only
        # one observation
        if len(cpp[0]) == 0:
            return

        tables = {}
        for ii in cpp[0]:
            tables[ii] = np.zeros((2, 2), dtype=np.float64)

        for i in range(self.model.num_group):

            endog_expval, _ = cached_means[i]

            emat_11 = self.get_eyy(endog_expval, i)
            emat_10 = endog_expval[:, None] - emat_11
            emat_01 = -emat_11 + endog_expval
            emat_00 = 1. - (emat_11 + emat_10 + emat_01)

            cpp1 = cpp[i]
            for ky in iterkeys(cpp1):
                ix = cpp1[ky]
                tables[ky][1, 1] += emat_11[ix[:, 0], ix[:, 1]].sum()
                tables[ky][1, 0] += emat_10[ix[:, 0], ix[:, 1]].sum()
                tables[ky][0, 1] += emat_01[ix[:, 0], ix[:, 1]].sum()
                tables[ky][0, 0] += emat_00[ix[:, 0], ix[:, 1]].sum()

        cor_expval = self.pooled_odds_ratio(list(itervalues(tables)))

        self.dep_params *= self.crude_or / cor_expval
        if not np.isfinite(self.dep_params):
            self.dep_params = 1.
            warnings.warn("dep_params became inf, resetting to 1",
                          ConvergenceWarning)

    update.__doc__ = CovStruct.update.__doc__
    covariance_matrix.__doc__ = CovStruct.covariance_matrix.__doc__

    def summary(self):

        return "Global odds ratio: %.3f\n" % self.dep_params


class OrdinalIndependence(CategoricalCovStruct):
    """
    An independence covariance structure for ordinal models.

    The working covariance between indicators derived from different
    observations is zero.  The working covariance between indicators
    derived form a common observation is determined from their current
    mean values.

    There are no parameters to estimate in this covariance structure.
    """

    def covariance_matrix(self, expected_value, index):

        ibd = self.ibd[index]
        n = len(expected_value)
        vmat = np.zeros((n, n))

        for bdl in ibd:
            ev = expected_value[bdl[0]:bdl[1]]
            vmat[bdl[0]:bdl[1], bdl[0]:bdl[1]] =\
                np.minimum.outer(ev, ev) - np.outer(ev, ev)

        return vmat, False

    # Nothing to update
    def update(self, params):
        pass


class NominalIndependence(CategoricalCovStruct):
    """
    An independence covariance structure for nominal models.

    The working covariance between indicators derived from different
    observations is zero.  The working covariance between indicators
    derived form a common observation is determined from their current
    mean values.

    There are no parameters to estimate in this covariance structure.
    """

    def covariance_matrix(self, expected_value, index):

        ibd = self.ibd[index]
        n = len(expected_value)
        vmat = np.zeros((n, n))

        for bdl in ibd:
            ev = expected_value[bdl[0]:bdl[1]]
            vmat[bdl[0]:bdl[1], bdl[0]:bdl[1]] =\
                np.diag(ev) - np.outer(ev, ev)

        return vmat, False

    # Nothing to update
    def update(self, params):
        pass


class Equivalence(CovStruct):
    """
    A covariance structure defined in terms of equivalence classes.

    An 'equivalence class' is a set of pairs of observations such that
    the covariance of every pair within the equivalence class has a
    common value.

    Parameters
    ----------
    pairs : dict-like
      A dictionary of dictionaries, where `pairs[group][label]`
      provides the indices of all pairs of observations in the group
      that have the same covariance value.  Specifically,
      `pairs[group][label]` is a tuple `(j1, j2)`, where `j1` and `j2`
      are integer arrays of the same length.  `j1[i], j2[i]` is one
      index pair that belongs to the `label` equivalence class.  Only
      one triangle of each covariance matrix should be included.
      Positions where j1 and j2 have the same value are variance
      parameters.
    labels : array-like
      An array of labels such that every distinct pair of labels
      defines an equivalence class.  Either `labels` or `pairs` must
      be provided.  When the two labels in a pair are equal two
      equivalence classes are defined: one for the diagonal elements
      (corresponding to variances) and one for the off-diagonal
      elements (corresponding to covariances).
    return_cov : boolean
      If True, `covariance_matrix` returns an estimate of the
      covariance matrix, otherwise returns an estimate of the
      correlation matrix.

    Notes
    -----
    Using `labels` to define the class is much easier than using
    `pairs`, but is less general.

    Any pair of values not contained in `pairs` will be assigned zero
    covariance.

    The index values in `pairs` are row indices into the `exog`
    matrix.  They are not updated if missing data are present.  When
    using this covariance structure, missing data should be removed
    before constructing the model.

    If using `labels`, after a model is defined using the covariance
    structure it is possible to remove a label pair from the second
    level of the `pairs` dictionary to force the corresponding
    covariance to be zero.

    Examples
    --------
    The following sets up the `pairs` dictionary for a model with two
    groups, equal variance for all observations, and constant
    covariance for all pairs of observations within each group.

    >> pairs = {0: {}, 1: {}}
    >> pairs[0][0] = (np.r_[0, 1, 2], np.r_[0, 1, 2])
    >> pairs[0][1] = np.tril_indices(3, -1)
    >> pairs[1][0] = (np.r_[3, 4, 5], np.r_[3, 4, 5])
    >> pairs[1][2] = 3 + np.tril_indices(3, -1)
    """

    def __init__(self, pairs=None, labels=None, return_cov=False):

        super(Equivalence, self).__init__()

        if (pairs is None) and (labels is None):
            raise ValueError(
                "Equivalence cov_struct requires either `pairs` or `labels`")

        if (pairs is not None) and (labels is not None):
            raise ValueError(
                "Equivalence cov_struct accepts only one of `pairs` "
                "and `labels`")

        if pairs is not None:
            import copy
            self.pairs = copy.deepcopy(pairs)

        if labels is not None:
            self.labels = np.asarray(labels)

        self.return_cov = return_cov

    def _make_pairs(self, i, j):
        """
        Create arrays containing all unique ordered pairs of i, j.

        The arrays i and j must be one-dimensional containing non-negative
        integers.
        """

        mat = np.zeros((len(i) * len(j), 2), dtype=np.int32)

        # Create the pairs and order them
        f = np.ones(len(j))
        mat[:, 0] = np.kron(f, i).astype(np.int32)
        f = np.ones(len(i))
        mat[:, 1] = np.kron(j, f).astype(np.int32)
        mat.sort(1)

        # Remove repeated rows
        try:
            dtype = np.dtype((np.void, mat.dtype.itemsize * mat.shape[1]))
            bmat = np.ascontiguousarray(mat).view(dtype)
            _, idx = np.unique(bmat, return_index=True)
        except TypeError:
            # workaround for old numpy that can't call unique with complex
            # dtypes
            rs = np.random.RandomState(4234)
            bmat = np.dot(mat, rs.uniform(size=mat.shape[1]))
            _, idx = np.unique(bmat, return_index=True)
        mat = mat[idx, :]

        return mat[:, 0], mat[:, 1]

    def _pairs_from_labels(self):

        from collections import defaultdict
        pairs = defaultdict(lambda: defaultdict(lambda: None))

        model = self.model

        df = pd.DataFrame({"labels": self.labels, "groups": model.groups})
        gb = df.groupby(["groups", "labels"])

        ulabels = np.unique(self.labels)

        for g_ix, g_lb in enumerate(model.group_labels):

            # Loop over label pairs
            for lx1 in range(len(ulabels)):
                for lx2 in range(lx1 + 1):

                    lb1 = ulabels[lx1]
                    lb2 = ulabels[lx2]

                    try:
                        i1 = gb.groups[(g_lb, lb1)]
                        i2 = gb.groups[(g_lb, lb2)]
                    except KeyError:
                        continue

                    i1, i2 = self._make_pairs(i1, i2)

                    clabel = str(lb1) + "/" + str(lb2)

                    # Variance parameters belong in their own equiv class.
                    jj = np.flatnonzero(i1 == i2)
                    if len(jj) > 0:
                        clabelv = clabel + "/v"
                        pairs[g_lb][clabelv] = (i1[jj], i2[jj])

                    # Covariance parameters
                    jj = np.flatnonzero(i1 != i2)
                    if len(jj) > 0:
                        i1 = i1[jj]
                        i2 = i2[jj]
                        pairs[g_lb][clabel] = (i1, i2)

        self.pairs = pairs

    def initialize(self, model):

        super(Equivalence, self).initialize(model)

        if self.model.weights is not None:
            warnings.warn("weights not implemented for equalence cov_struct, "
                          "using unweighted covariance estimate",
                          NotImplementedWarning)

        if not hasattr(self, 'pairs'):
            self._pairs_from_labels()

        # Initialize so that any equivalence class containing a
        # variance parameter has value 1.
        self.dep_params = defaultdict(lambda: 0.)
        self._var_classes = set([])
        for gp in self.model.group_labels:
            for lb in self.pairs[gp]:
                j1, j2 = self.pairs[gp][lb]
                if np.any(j1 == j2):
                    if not np.all(j1 == j2):
                        warnings.warn(
                            "equivalence class contains both variance "
                            "and covariance parameters", OutputWarning)
                    self._var_classes.add(lb)
                    self.dep_params[lb] = 1

        # Need to start indexing at 0 within each group.
        # rx maps olds indices to new indices
        rx = -1 * np.ones(len(self.model.endog), dtype=np.int32)
        for g_ix, g_lb in enumerate(self.model.group_labels):
            ii = self.model.group_indices[g_lb]
            rx[ii] = np.arange(len(ii), dtype=np.int32)

        # Reindex
        for gp in self.model.group_labels:
            for lb in self.pairs[gp].keys():
                a, b = self.pairs[gp][lb]
                self.pairs[gp][lb] = (rx[a], rx[b])

    def update(self, params):

        endog = self.model.endog_li
        varfunc = self.model.family.variance
        cached_means = self.model.cached_means
        dep_params = defaultdict(lambda: [0., 0., 0.])
        n_pairs = defaultdict(lambda: 0)
        dim = len(params)

        for k, gp in enumerate(self.model.group_labels):
            expval, _ = cached_means[k]
            stdev = np.sqrt(varfunc(expval))
            resid = (endog[k] - expval) / stdev
            for lb in self.pairs[gp].keys():
                if (not self.return_cov) and lb in self._var_classes:
                    continue
                jj = self.pairs[gp][lb]
                dep_params[lb][0] += np.sum(resid[jj[0]] * resid[jj[1]])
                if not self.return_cov:
                    dep_params[lb][1] += np.sum(resid[jj[0]] ** 2)
                    dep_params[lb][2] += np.sum(resid[jj[1]] ** 2)
                n_pairs[lb] += len(jj[0])

        if self.return_cov:
            for lb in dep_params.keys():
                dep_params[lb] = dep_params[lb][0] / (n_pairs[lb] - dim)
        else:
            for lb in dep_params.keys():
                den = np.sqrt(dep_params[lb][1] * dep_params[lb][2])
                dep_params[lb] = dep_params[lb][0] / den
            for lb in self._var_classes:
                dep_params[lb] = 1.

        self.dep_params = dep_params
        self.n_pairs = n_pairs

    def covariance_matrix(self, expval, index):
        dim = len(expval)
        cmat = np.zeros((dim, dim))
        g_lb = self.model.group_labels[index]

        for lb in self.pairs[g_lb].keys():
            j1, j2 = self.pairs[g_lb][lb]
            cmat[j1, j2] = self.dep_params[lb]

        cmat = cmat + cmat.T
        np.fill_diagonal(cmat, cmat.diagonal() / 2)

        return cmat, not self.return_cov

    update.__doc__ = CovStruct.update.__doc__
    covariance_matrix.__doc__ = CovStruct.covariance_matrix.__doc__
