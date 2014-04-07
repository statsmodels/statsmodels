import numpy as np


class CovStruct(object):
    """
    The base class for correlation and covariance structures of
    cluster data.

    Each implementation of this class takes the residuals from a
    regression model that has been fitted to clustered data, and uses
    them to estimate the within-cluster variance and dependence
    structure of the random errors in the model.
    """

    # Parameters describing the dependency structure
    dep_params = None


    def initialize(self, parent):
        """
        Called by GEE, used by implementations that need additional
        setup prior to running `fit`.

        Parameters
        ----------
        parent : GEE class
            A reference to the parent GEE class instance.
        """
        pass


    def update(self, beta, parent):
        """
        Updates the association parameter values based on the current
        regression coefficients.

        Parameters
        ----------
        beta : array-like
            Working values for the regression parameters.
        parent : GEE class reference
            A reference to the parent GEE class instance.
        """
        raise NotImplementedError


    def covariance_matrix(self, endog_expval, index):
        """Returns the working covariance or correlation matrix for a
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

    dep_params = np.r_[[]]

    # Nothing to update
    def update(self, beta, parent):
        return


    def covariance_matrix(self, expval, index):
        dim = len(expval)
        return np.eye(dim, dtype=np.float64), True


    def summary(self):
        return "Observations within a cluster are independent."


class Exchangeable(CovStruct):
    """
    An exchangeable working dependence structure.
    """

    # The correlation between any two values in the same cluster
    dep_params = 0


    def update(self, beta, parent):

        endog = parent.endog_li

        num_clust = len(endog)
        nobs = parent.nobs
        dim = len(beta)

        varfunc = parent.family.variance

        cached_means = parent.cached_means

        residsq_sum, scale, nterm = 0, 0, 0
        for i in range(num_clust):

            if len(endog[i]) == 0:
                continue

            expval, _ = cached_means[i]

            sdev = np.sqrt(varfunc(expval))
            resid = (endog[i] - expval) / sdev

            ngrp = len(resid)
            residsq = np.outer(resid, resid)
            scale += np.diag(residsq).sum()
            residsq = np.tril(residsq, -1)
            residsq_sum += residsq.sum()
            nterm += 0.5 * ngrp * (ngrp - 1)

        scale /= (nobs - dim)
        self.dep_params = residsq_sum / (scale * (nterm - dim))


    def covariance_matrix(self, expval, index):
        dim = len(expval)
        dp = self.dep_params * np.ones((dim, dim), dtype=np.float64)
        return  dp + (1 - self.dep_params) * np.eye(dim), True

    def summary(self):
        return ("The correlation between two observations in the " +
                "same cluster is %.3f" % self.dep_params)




class Nested(CovStruct):
    """A nested working dependence structure.

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

    Example
    -------
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

    # The regression design matrix for estimating the variance
    # components
    designx = None

    # Matrices containing labels that indicate which covariance
    # parameters are constrained to be equal
    ilabels = None

    # The SVD of designx
    designx_u = None
    designx_s = None
    designx_v = None

    # The scale parameter
    scale = None

    # The regression coefficients for estimating the variance
    # components
    vcomp_coeff = None



    def initialize(self, parent):
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

        # A bit of processing of the nest data
        id_matrix = np.asarray(parent.dep_data)
        if id_matrix.ndim == 1:
            id_matrix = id_matrix[:,None]
        self.id_matrix = id_matrix

        endog = parent.endog_li
        num_clust = len(endog)
        designx, ilabels = [], []

        # The number of layers of nesting
        n_nest = self.id_matrix.shape[1]

        for i in range(num_clust):
            ngrp = len(endog[i])
            glab = parent.group_labels[i]
            rix = parent.group_indices[glab]

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
            dsx = np.zeros((len(ix1), n_nest+1), dtype=np.float64)
            dsx[:,0] = 1
            for k in np.unique(ncm):
                ii = np.flatnonzero(ncm == k)
                dsx[ii, 1:k+1] = 1
            designx.append(dsx)

        self.designx = np.concatenate(designx, axis=0)
        self.ilabels = ilabels

        svd = np.linalg.svd(self.designx, 0)
        self.designx_u = svd[0]
        self.designx_s = svd[1]
        self.designx_v = svd[2].T


    def update(self, beta, parent):

        endog = parent.endog_li
        offset = parent.offset_li

        num_clust = len(endog)
        nobs = parent.nobs
        dim = len(beta)

        if self.designx is None:
            self._compute_design(parent)

        cached_means = parent.cached_means

        varfunc = parent.family.variance

        dvmat = []
        scale = 0.
        for i in range(num_clust):

            # Needed?
            if len(endog[i]) == 0:
                continue

            expval, _ = cached_means[i]

            sdev = np.sqrt(varfunc(expval))
            resid = (endog[i] - offset[i] - expval) / sdev

            ix1, ix2 = np.tril_indices(len(resid), -1)
            dvmat.append(resid[ix1] * resid[ix2])

            scale += np.sum(resid**2)

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


    def summary(self):
        """
        Returns a summary string describing the state of the
        dependence structure.
        """

        msg = "Variance estimates\n------------------\n"
        for k in range(len(self.vcomp_coeff)):
            msg += "Component %d: %.3f\n" % (k+1, self.vcomp_coeff[k])
        msg += "Residual: %.3f\n" % (self.scale -
                                     np.sum(self.vcomp_coeff))
        return msg



class Autoregressive(CovStruct):
    """
    An autoregressive working dependence structure.  The dependence is
    defined in terms of the `time` component of the parent GEE class.
    Time represents a potentially multidimensional index from which
    distances between pairs of obsercations can be determined.  The
    correlation between two observations in the same cluster is
    dep_params**distance, where `dep_params` is the autocorrelation
    parameter to be estimated, and distance is the distance between
    the two observations, calculated from their corresponding time
    values.  `time` is stored as an n_obs x k matrix, where `k`
    represents the number of dimensions in the time index.

    The autocorrelation parameter is estimated using weighted
    nonlinear least squares, regressing each value within a cluster on
    each preceeding value within the same cluster.

    Parameters
    ----------
    dist_func: function from R^k x R^k to R^+, optional
       A function that takes the time vector for two observations and
       computed the distance between the two observations based on the
       time vector.

    Reference
    ---------
    B Rosner, A Munoz.  Autoregressive modeling for the analysis of
    longitudinal data with unequally spaced examinations.  Statistics
    in medicine. Vol 7, 59-71, 1988.
    """

    # The autoregression parameter
    dep_params = 0

    designx = None

    # The function for determining distances based on time
    dist_func = None


    def __init__(self, dist_func=None):

        if dist_func is None:
            self.dist_func = lambda x, y: np.abs(x - y).sum()
        else:
            self.dist_func = dist_func


    def update(self, beta, parent):

        if parent.time is None:
            raise ValueError("GEE: time must be provided to GEE if "
                             "using AR dependence structure")

        endog = parent.endog_li
        time = parent.time_li

        num_clust = len(endog)

        # Only need to compute this once
        if self.designx is not None:
            designx = self.designx
        else:
            designx = []
            for i in range(num_clust):

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

        scale = parent.estimate_scale()

        varfunc = parent.family.variance

        cached_means = parent.cached_means

        # Weights
        var = (1 - self.dep_params**(2 * designx)) /\
            (1 - self.dep_params**2)
        wts = 1 / var
        wts /= wts.sum()

        residmat = []
        for i in range(num_clust):

            if len(endog[i]) == 0:
                continue

            expval, _ = cached_means[i]

            sdev = np.sqrt(scale * varfunc(expval))
            resid = (endog[i] - expval) / sdev

            ngrp = len(resid)
            for j1 in range(ngrp):
                for j2 in range(j1):
                    residmat.append([resid[j1], resid[j2]])

        residmat = np.array(residmat)

        # Need to minimize this
        def fitfunc(a):
            dif = residmat[:, 0] - (a**designx)*residmat[:, 1]
            return np.dot(dif**2, wts)

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
            b_rgt = b_rgt + (1 - b_rgt) / 2
            f_rgt = fitfunc(b_rgt)
            if b_rgt > 1 - 1e-6:
                raise ValueError(
                    "Autoregressive: unable to find right bracket")

        from scipy.optimize import brent
        self.dep_params = brent(fitfunc, brack=[b_lft, b_ctr, b_rgt])


    def covariance_matrix(self, endog_expval, index):
        ngrp = len(endog_expval)
        if self.dep_params == 0:
            return np.eye(ngrp, dtype=np.float64), True
        idx = np.arange(ngrp)
        cmat = self.dep_params**np.abs(idx[:, None] - idx[None, :])
        return cmat, True


    def summary(self):

        return ("Autoregressive(1) dependence parameter: %.3f\n" %
                self.dep_params)



class GlobalOddsRatio(CovStruct):
    """
    Estimate the global odds ratio for a GEE with ordinal or nominal
    data.

    References
    ----------
    PJ Heagerty and S Zeger. "Marginal Regression Models for Clustered
    Ordinal Measurements". Journal of the American Statistical
    Association Vol. 91, Issue 435 (1996).

    Generalized Estimating Equations for Ordinal Data: A Note on
    Working Correlation Structures Thomas Lumley Biometrics Vol. 52,
    No. 1 (Mar., 1996), pp. 354-361
    http://www.jstor.org/stable/2533173

    Notes:
    ------
    'ibd' is a list whose i^th element ibd[i] is a sequence of tuples
    (a,b), where endog[i][a:b] is the subvector of indicators derived
    from the same ordinal value.

    `cpp` is a dictionary where cpp{group} is a map from cut-point
    pairs (c,c') to the indices of between-subject pairs derived from
    the given cut points.

    """

    # The current estimate of the odds ratio
    dep_params = [None,]

    # The current estimate of the crude odds ratio
    crude_or = None

    # See docstring
    ibd = None

    # See docstring
    cpp = None


    def __init__(self, nlevel, endog_type):
        super(GlobalOddsRatio, self).__init__()
        self.nlevel = nlevel
        self.ncut = nlevel - 1
        self.endog_type = endog_type


    def initialize(self, parent):

        ibd = []
        for v in parent.endog_li:
            jj = np.arange(0, len(v) + 1, self.ncut)
            ibd1 = np.hstack((jj[0:-1][:, None], jj[1:][:, None]))
            ibd1 = [(jj[k], jj[k + 1]) for k in range(len(jj) - 1)]
            ibd.append(ibd1)
        self.ibd = ibd

        cpp = []
        for v in parent.endog_li:
            m = len(v) / self.ncut
            jj = np.kron(np.ones(m), np.arange(self.ncut))
            j1 = np.outer(jj, np.ones(len(jj)))
            j2 = np.outer(np.ones(len(jj)), jj)
            cpp1 = {}
            for k1 in range(self.ncut):
                for k2 in range(k1+1):
                    v1, v2 = np.nonzero((j1==k1) & (j2==k2))
                    cpp1[(k2, k1)] = \
                        np.hstack((v2[:, None], v1[:, None]))
            cpp.append(cpp1)
        self.cpp = cpp

        # Initialize the dependence parameters
        self.crude_or = self.observed_crude_oddsratio(parent)
        self.dep_params[0] = self.crude_or


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
        log_pooled_or = sum([w*e for w, e in zip(wts, log_oddsratio)])

        return np.exp(log_pooled_or)


    def covariance_matrix(self, expected_value, index):

        vmat = self.get_eyy(expected_value, index)
        vmat -= np.outer(expected_value, expected_value)
        return vmat, False


    def observed_crude_oddsratio(self, parent):
        """The crude odds ratio is obtained by pooling all data
        corresponding to a given pair of cut points (c,c'), then
        forming the inverse variance weighted average of these odds
        ratios to obtain a single OR.  Since the covariate effects are
        ignored, this OR will generally be greater than the stratified
        OR.
        """

        cpp = self.cpp
        endog = parent.endog_li

        # Storage for the contingency tables for each (c,c')
        tables = {}
        for ii in cpp[0].keys():
            tables[ii] = np.zeros((2, 2), dtype=np.float64)

        # Get the observed crude OR
        for i in range(len(endog)):

            if len(endog[i]) == 0:
                continue

            # The observed joint values for the current cluster
            yvec = endog[i]
            endog_11 = np.outer(yvec, yvec)
            endog_10 = np.outer(yvec, 1 - yvec)
            endog_01 = np.outer(1 - yvec, yvec)
            endog_00 = np.outer(1 - yvec, 1 - yvec)

            cpp1 = cpp[i]
            for ky in cpp1.keys():
                ix = cpp1[ky]
                tables[ky][1, 1] += endog_11[ix[:, 0], ix[:, 1]].sum()
                tables[ky][1, 0] += endog_10[ix[:, 0], ix[:, 1]].sum()
                tables[ky][0, 1] += endog_01[ix[:, 0], ix[:, 1]].sum()
                tables[ky][0, 0] += endog_00[ix[:, 0], ix[:, 1]].sum()

        return self.pooled_odds_ratio(tables.values())



    def get_eyy(self, endog_expval, index):
        """
        Returns a matrix V such that V[i,j] is the joint probability
        that endog[i] = 1 and endog[j] = 1, based on the marginal
        probabilities of endog and the odds ratio cor.
        """

        cor = self.dep_params[0]
        ibd = self.ibd[index]

        # The between-observation joint probabilities
        if cor == 1.0:
            vmat = np.outer(endog_expval, endog_expval)
        else:
            psum = endog_expval[:, None] + endog_expval[None, :]
            pprod = endog_expval[:, None] * endog_expval[None, :]
            pfac = np.sqrt((1 + psum * (cor-1))**2 +
                           4 * cor * (1 - cor) * pprod)
            vmat = 1 +  psum * (cor - 1) - pfac
            vmat /= 2 * (cor - 1)

        # Fix E[YY'] for elements that belong to same observation
        for bdl in ibd:
            evy = endog_expval[bdl[0]:bdl[1]]
            if self.endog_type == "ordinal":
                eyr = np.outer(evy, np.ones(len(evy)))
                eyc = np.outer(np.ones(len(evy)), evy)
                vmat[bdl[0]:bdl[1], bdl[0]:bdl[1]] = \
                    np.where(eyr < eyc, eyr, eyc)
            else:
                vmat[bdl[0]:bdl[1], bdl[0]:bdl[1]] = np.diag(evy)

        return vmat


    def update(self, beta, parent):
        """Update the global odds ratio based on the current value of
        beta."""

        endog = parent.endog_li
        cpp = self.cpp
        cached_means = parent.cached_means

        num_clust = len(endog)

        # This will happen if all the clusters have only
        # one observation
        if len(cpp[0]) == 0:
            return

        tables = {}
        for ii in cpp[0]:
            tables[ii] = np.zeros((2, 2), dtype=np.float64)

        for i in range(num_clust):

            if len(endog[i]) == 0:
                continue

            endog_expval, _ = cached_means[i]

            emat_11 = self.get_eyy(endog_expval, i)
            emat_10 = endog_expval[:, None] - emat_11
            emat_01 = -emat_11 + endog_expval
            emat_00 = 1 - (emat_11 + emat_10 + emat_01)

            cpp1 = cpp[i]
            for ky in cpp1.keys():
                ix = cpp1[ky]
                tables[ky][1, 1] += emat_11[ix[:, 0], ix[:, 1]].sum()
                tables[ky][1, 0] += emat_10[ix[:, 0], ix[:, 1]].sum()
                tables[ky][0, 1] += emat_01[ix[:, 0], ix[:, 1]].sum()
                tables[ky][0, 0] += emat_00[ix[:, 0], ix[:, 1]].sum()

        cor_expval = self.pooled_odds_ratio(tables.values())

        self.dep_params[0] *= self.crude_or / cor_expval


    def summary(self):

        return "Global odds ratio: %.3f\n" % self.dep_params[0]
