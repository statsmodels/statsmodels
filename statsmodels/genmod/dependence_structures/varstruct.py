import numpy as np


class VarStruct(object):
    """
    A base class for correlation and covariance structures of repeated
    measures data.  Each implementation of this class takes the
    residuals from a regression fit to clustered data, and uses the
    residuals from the fit to estimate the within-cluster variance and
    dependence structure of the model errors.
    """

    # The parent model instance
    parent = None

    # Parameters describing the dependency structure
    dparams = None


    def initialize(self, parent):
        """
        Parameters
        ----------
        parent : a reference to the model using this dependence
        structure

        Notes
        -----
        The clustered data should be availabe as `parent.endog` and
        `parent.exog`, where `endog` and `exog` are lists of the same
        length.  `endog[i]` is the response data represented as a n_i
        length ndarray, and `endog[i]` is the covariate data
        represented as a n_i x p ndarray, where n_i is the number of
        observations in cluster i.
        """
        self.parent = parent


    def update(self, beta):
        """
        Updates the association parameter values based on the current
        regression coefficients.
        """
        raise NotImplementedError


    def variance_matrix(self, endog_expval, index):
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


class Independence(VarStruct):
    """
    An independence working dependence structure.
    """

    # Nothing to update
    def update(self, beta):
        return


    def variance_matrix(self, expval, index):
        dim = len(expval)
        return np.eye(dim, dtype=np.float64), True


    def summary(self):
        return "Observations within a cluster are independent."


class Exchangeable(VarStruct):
    """
    An exchangeable working dependence structure.
    """

    # The correlation between any two values in the same cluster
    dparams = 0


    def update(self, beta):

        endog = self.parent.endog_li

        num_clust = len(endog)
        nobs = self.parent.nobs
        dim = len(beta)

        varfunc = self.parent.family.variance

        cached_means = self.parent.cached_means

        residsq_sum, scale_inv, nterm = 0, 0, 0
        for i in range(num_clust):

            if len(endog[i]) == 0:
                continue

            expval, _ = cached_means[i]

            sdev = np.sqrt(varfunc(expval))
            resid = (endog[i] - expval) / sdev

            ngrp = len(resid)
            residsq = np.outer(resid, resid)
            scale_inv += np.diag(residsq).sum()
            residsq = np.tril(residsq, -1)
            residsq_sum += residsq.sum()
            nterm += 0.5 * ngrp * (ngrp - 1)

        scale_inv /= (nobs - dim)
        self.dparams = residsq_sum / (scale_inv * (nterm - dim))


    def variance_matrix(self, expval, index):
        dim = len(expval)
        return self.dparams * np.ones((dim, dim), dtype=np.float64) + \
                                (1 - self.dparams) * np.eye(dim), True

    def summary(self):
        return "The correlation between two observations in the same cluster is %.3f" % self.dparams




class Nested(VarStruct):
    """A nested working dependence structure.

    The variance components are estimated using least squares
    regression of the products y*y', for outcomes y and y' in the same
    cluster, on a vector of indicators defining which variance
    components are shared by y and y'.

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

    # The inverse of the scale parameter
    scale_inv = None

    # The regression coefficients for estimating the variance
    # components
    vcomp_coeff = None


    def __init__(self, id_matrix):
        """
        A working dependence structure that captures a nested sequence
        of clusters.

        Parameters
        ----------
        id_matrix : array-like
           An n_obs x k matrix of cluster indicators, corresponding to
           clusters nested under the top-level clusters provided to
           GEE.  These clusters should be nested from left to right,
           so that two observations with the same value for column j
           of Id should also have the same value for cluster j' < j of
           Id (this only applies to observations in the same top-level
           cluster).

        Notes
        -----
        Suppose our data are student test scores, and the students are
        in in classrooms, nested in schools, nested in school
        districts.  Then the school district id would be provided to
        GEE as the top-level cluster assignment, and the school and
        classroom id's would be provided to the instance of the Nested
        class, for example

        0 0  School 0, classroom 0
        0 0  School 0, classroom 0
        0 1  School 0, classroom 1
        0 1  School 0, classroom 1
        1 0  School 1, classroom 0
        1 0  School 1, classroom 0
        1 1  School 1, classroom 1
        1 1  School 1, classroom 1
        """

        # A bit of processing of the Id argument
        if type(id_matrix) != np.ndarray:
            id_matrix = np.array(id_matrix)
        if len(id_matrix.shape) == 1:
            id_matrix = id_matrix[:, None]
        self.id_matrix = id_matrix

        # To be defined on the first call to update
        self.designx = None


    def _compute_design(self):
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

        endog = self.parent.endog_li
        num_clust = len(endog)
        designx, ilabels = [], []
        n_nest = self.id_matrix.shape[1]
        for i in range(num_clust):
            ngrp = len(endog[i])
            rix = self.parent.row_indices[i]

            ilabel = np.zeros((ngrp, ngrp), dtype=np.int32)
            for j1 in range(ngrp):
                for j2 in range(j1):

                    # Number of common nests.
                    ncm = np.sum(self.id_matrix[rix[j1], :] ==
                                 self.id_matrix[rix[j2], :])

                    dsx = np.zeros(n_nest+1, dtype=np.float64)
                    dsx[0] = 1
                    dsx[1:ncm+1] = 1
                    designx.append(dsx)
                    ilabel[j1, j2] = ncm + 1
                    ilabel[j2, j1] = ncm + 1
            ilabels.append(ilabel)
        self.designx = np.array(designx)
        self.ilabels = ilabels

        svd = np.linalg.svd(self.designx, 0)
        self.designx_u = svd[0]
        self.designx_s = svd[1]
        self.designx_v = svd[2].T


    def update(self, beta):

        endog = self.parent.endog_li
        offset = self.parent.offset_li

        num_clust = len(endog)
        nobs = self.parent.nobs
        dim = len(beta)

        if self.designx is None:
            self._compute_design()

        cached_means = self.parent.cached_means

        varfunc = self.parent.family.variance

        dvmat = []
        scale_inv = 0.
        for i in range(num_clust):

            if len(endog[i]) == 0:
                continue

            expval, _ = cached_means[i]

            sdev = np.sqrt(varfunc(expval))
            resid = (endog[i] - offset[i] - expval) / sdev

            ngrp = len(resid)
            for j1 in range(ngrp):
                for j2 in range(j1):
                    dvmat.append(resid[j1] * resid[j2])

            scale_inv += np.sum(resid**2)

        dvmat = np.array(dvmat)
        scale_inv /= (nobs - dim)

        # Use least squares regression to estimate the variance
        # components
        vcomp_coeff = np.dot(self.designx_v, np.dot(self.designx_u.T, dvmat) /\
                                 self.designx_s)

        self.vcomp_coeff = np.clip(vcomp_coeff, 0, np.inf)
        self.scale_inv = scale_inv

        self.dparams = self.vcomp_coeff.copy()


    def variance_matrix(self, expval, index):

        dim = len(expval)

        # First iteration
        if self.designx is None:
            return np.eye(dim, dtype=np.float64), True

        ilabel = self.ilabels[index]

        c = np.r_[self.scale_inv, np.cumsum(self.vcomp_coeff)]
        vmat = c[ilabel]
        vmat /= self.scale_inv
        return vmat, True


    def summary(self):

        msg = "Variance estimates\n------------------\n"
        for k in range(len(self.vcomp_coeff)):
            msg += "Component %d: %.3f\n" % (k+1, self.vcomp_coeff[k])
        msg += "Residual: %.3f\n" % (self.scale_inv - np.sum(self.vcomp_coeff))
        return msg



class Autoregressive(VarStruct):
    """
    An autoregressive working dependence structure.  The dependence is
    defined in terms of the `time` component of the parent GEE class.
    Time represents a potentially multidimensional index from which
    distances between pairs of obsercations can be determined.  The
    correlation between two observations in the same cluster is
    dparams**distance, where `dparams` is the autocorrelation
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
    dparams = 0

    designx = None

    # The function for determining distances based on time
    dist_func = None


    def __init__(self, dist_func=None):

        if dist_func is None:
            self.dist_func = lambda x, y: np.abs(x - y).sum()
        else:
            self.dist_func = dist_func


    def update(self, beta):

        if self.parent.time is None:
            raise ValueError("GEE: time must be provided to GEE if "
                             "using AR dependence structure")

        endog = self.parent.endog_li
        time = self.parent.time_li

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

        scale = self.parent.estimate_scale()

        varfunc = self.parent.family.variance

        cached_means = self.parent.cached_means

        # Weights
        var = (1 - self.dparams**(2 * designx)) / (1 - self.dparams**2)
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
                self.dparams = 0
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
        self.dparams = brent(fitfunc, brack=[b_lft, b_ctr, b_rgt])


    def variance_matrix(self, endog_expval, index):
        ngrp = len(endog_expval)
        if self.dparams == 0:
            return np.eye(ngrp, dtype=np.float64), True
        idx = np.arange(ngrp)
        return self.dparams**np.abs(idx[:, None] - idx[None, :]), True


    def summary(self):

        print "Autoregressive(1) dependence parameter: %.3f\n" % self.dparams



class GlobalOddsRatio(VarStruct):
    """
    Estimate the global `qodds ratio for a GEE with either ordinal or
    nominal data.

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
    odds_ratio = None

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

        self.parent = parent

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
        self.crude_or = self.observed_crude_oddsratio()
        self.odds_ratio = self.crude_or


    def pooled_odds_ratio(self, tables):
        """
        Returns the pooled odds ratio for a list of 2x2 tables.

        The pooled odds ratio is the inverse variance weighted average
        of the sample odds ratios of the tables.
        """

        if len(tables) == 0:
            return 1.

        # Get the samepled odds ratios and variances
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


    def variance_matrix(self, expected_value, index):

        vmat = self.get_eyy(expected_value, index)
        vmat -= np.outer(expected_value, expected_value)
        return vmat, False


    def observed_crude_oddsratio(self):
        """The crude odds ratio is obtained by pooling all data
        corresponding to a given pair of cut points (c,c'), then
        forming the inverse variance weighted average of these odds
        ratios to obtain a single OR.  Since the covariate effects are
        ignored, this OR will generally be greater than the stratified
        OR.
        """

        cpp = self.cpp
        endog = self.parent.endog_li

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

        cor = self.odds_ratio
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


    def update(self, beta):
        """Update the global odds ratio based on the current value of
        beta."""

        endog = self.parent.endog_li
        cpp = self.cpp
        cached_means = self.parent.cached_means

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

        self.odds_ratio *= self.crude_or / cor_expval


    def summary(self):

        print "Global odds ratio: %.3f\n" % self.odds_ratio
