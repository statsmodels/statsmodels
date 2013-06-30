import numpy as np


class VarStruct(object):
    """
    A base class for correlation and covariance structures of repeated
    measures data.
    """

    def initialize(self, parent):
        """
        Parameters
        ----------
        parent : a reference to the model using this dependence structure

        Notes
        -----
        parent should contain the clustered data in the form Y, X, where
        Y and X are lists of the same length.  Y[i] is the response data
        represented as a n_i length ndarray, and X[i] is the covariate
        data represented as a n_i x p ndarray, where n_i is the number of
        observations in cluster i.
        """
        self.parent = parent


    def update(self, beta):
        """
        Updates the association parameter values based on the current
        regression coefficients.
        """
        raise NotImplementedError


    def variance_matrix(self, E, index):
        """Returns the working covariance or correlation matrix for a given
        cluster of data.

        Parameters
        ----------
        E: array-like
           The expected values of Y for the cluster for which the covariance or
           correlation matrix will be returned
        index: integer
           The index of the cluster for which the covariane or correlation
           matrix will be returned

        Returns
        -------
        M: matrix
            The covariance or correlation matrix of Y
        is_cor: bool
            True if M is a correlation matrix, False if M is a covariance matrix
        """
        raise NotImplementedError


    def summary(self):
        """
        Returns a text summary of the current estimate of the dependence structure.
        """
        raise NotImplementedError


class Independence(VarStruct):
    """
    An independence working dependence structure.
    """

    # Nothing to update
    def update(self, beta):
        return


    def variance_matrix(self, E, index):
        n = len(E)
        return np.eye(n, dtype=np.float64),True


    def summary(self):
        return "Observations within a cluster are independent."


class Exchangeable(VarStruct):
    """
    An exchangeable working dependence structure.
    """

    # The correlation between any two values in the same cluster
    a = 0


    def update(self, beta):

        N = len(self.parent.Y)
        nobs = self.parent.nobs
        p = len(beta)

        mean = self.parent.family.link.inverse
        varfunc = self.parent.family.variance
        Y = self.parent.Y
        X = self.parent.X

        a,scale_inv,m = 0,0,0
        for i in range(N):

            if len(Y[i]) == 0:
                continue

            lp = np.dot(X[i], beta)
            E = mean(lp)

            S = np.sqrt(varfunc(E))
            resid = (self.parent.Y[i] - E) / S

            n = len(resid)
            Q = np.outer(resid, resid)
            scale_inv += np.diag(Q).sum()
            Q = np.tril(Q, -1)
            a += Q.sum()
            m += 0.5*n*(n-1)

        scale_inv /= (nobs-p)
        self.a = a/(scale_inv*(m-p))


    def variance_matrix(self, E, index):
        n = len(E)
        return self.a*np.ones((n,n), dtype=np.float64) + (1-self.a)*np.eye(n),True

    def summary(self):
        return "The correlation between two observations in the same cluster is %.3f" % self.a




class Nested(VarStruct):
    """A nested working dependence structure.

    The variance components are estimated using least squares
    regression of the products y*y', for outcomes y and y' in the same
    cluster, on a vector of indicators defining which variance
    components are shared by y and y'.

    """


    def __init__(self, Id):
        """
        A working dependence structure that captures a nested sequence of clusters.

        Parameters
        ----------
        Id : array-like
           An n_obs x k matrix of cluster indicators, corresponding to clusters
           nested under the top-level clusters provided to GEE.  These clusters
           should be nested from left to right, so that two observations with the
           same value for column j of Id should also have the same value for cluster
           j' < j of Id (this only applies to observations in the same top-level cluster).

        Notes
        -----
        Suppose our data are student test scores, and the students are in in classrooms,
        nested in schools, nested in school districts.  Then the school district Id would
        be provided to GEE as the top-level cluster assignment, and the school and classroom
        Id's would be provided to the instance of the Nested class, for example

        0 0  School 0, classroom 0
        0 0  School 0, classroom 0
        0 1  School 0, classroom 1
        0 1  School 0, classroom 1
        1 0  School 1, classroom 0 (not the same as classroom 0 in school 0)
        1 0  School 1, classroom 0
        1 1  School 1, classroom 1
        1 1  School 1, classroom 1
        """

        # A bit of processing of the Id argument
        if type(Id) != np.ndarray:
            Id = np.array(Id)
        if len(Id.shape) == 1:
            Id = Id[:,None]
        self.Id = Id

        # To be defined on the first call to update
        self.QX = None


    def _compute_design(self):
        """Called on the first call to update

        QI is a list of n_i x n_i matrices containing integer labels
        that correspond to specific correlation parameters.  Two
        elements of QI[i] with the same label share identical variance
        components.

        QX is a matrix, with each row containing dummy variables
        indicating which variance components are associated with the
        corresponding element of QY.
        """

        N = len(self.parent.Y)
        QX,QI = [],[]
        m = self.Id.shape[1]
        for i in range(N):
            n = len(self.parent.Y[i])
            ix = self.parent.IX[i]

            qi = np.zeros((n,n), dtype=np.int32)
            for j1 in range(n):
                for j2 in range(j1):
                    i1 = ix[j1]
                    i2 = ix[j2]
                    k = np.sum(self.Id[i1,:] == self.Id[i2,:])
                    x = np.zeros(m+1, dtype=np.float64)
                    x[0] = 1
                    x[1:k+1] = 1
                    QX.append(x)
                    qi[j1,j2] = k + 1
                    qi[j2,j1] = k + 1
            QI.append(qi)
        self.QX = np.array(QX)
        self.QI = QI

        u,s,vt = np.linalg.svd(self.QX, 0)
        self.QX_u = u
        self.QX_s = s
        self.QX_v = vt.T


    def update(self, beta):

        N = len(self.parent.Y)
        nobs = self.parent.nobs
        p = len(beta)

        if self.QX is None:
            self._compute_design()

        mean = self.parent.family.link.inverse
        varfunc = self.parent.family.variance
        Y = self.parent.Y
        X = self.parent.X

        QY = []
        scale_inv,m = 0.,0.
        for i in range(N):

            if len(Y[i]) == 0:
                continue

            lp = np.dot(X[i], beta)
            E = mean(lp)

            S = np.sqrt(varfunc(E))
            resid = (self.parent.Y[i] - E)/S

            n = len(resid)
            for j1 in range(n):
                for j2 in range(j1):
                    QY.append(resid[j1]*resid[j2])

            scale_inv += np.sum(resid**2)
            m += 0.5*n*(n-1)

        QY = np.array(QY)
        scale_inv /= (nobs-p)

        # Use least squares regression to estimate the variance components
        b = np.dot(self.QX_v, np.dot(self.QX_u.T, QY)/self.QX_s)

        self.b = np.clip(b, 0, np.inf)
        self.scale_inv = scale_inv


    def variance_matrix(self, E, index):

        n = len(E)

        # First iteration
        if self.QX is None:
            return np.eye(n, dtype=np.float64),True

        qi = self.QI[index]

        c = np.r_[self.scale_inv, np.cumsum(self.b)]
        C = c[qi]
        C /= self.scale_inv
        return C,True


    def summary(self):

        s = "Variance estimates\n------------------\n"
        for k in range(len(self.b)):
            s += "Component %d: %.3f\n" % (k+1, self.b[k])
        s += "Residual: %.3f\n" % (self.scale_inv - np.sum(self.b))
        return s



class Autoregressive(VarStruct):
    """
    An autoregressive working dependence structure.

    The autocorrelation parameter is estimated using weighted nonlinear
    least squares, regressing each value within a cluster on each preceeding
    value within the same cluster.

    Reference
    ---------
    B Rosner, A Munoz.  Autoregressive modeling for the analysis of
    longitudinal data with unequally spaced examinations.  Statistics
    in medicine. Vol 7, 59-71, 1988.
    """

    # The autoregression parameter
    a = 0

    QX = None

    def update(self, beta):

        if self.parent.T is None:
            raise ValueError("GEE: T must be provided to GEE if using AR dependence structure")

        N = len(self.parent.Y)
        nobs = self.parent.nobs
        p = len(beta)
        Y = self.parent.Y
        X = self.parent.X
        T = self.parent.T

        # Only need to compute this once
        if self.QX is not None:
            QX = self.QX
        else:
            QX = []
            for i in range(N):

                n = len(Y[i])
                if n == 0:
                    continue

                for j1 in range(n):
                    for j2 in range(j1):
                        QX.append(np.abs(T[i][j1] - T[i][j2]))

            QX = np.array(QX)
            self.QX = QX

        scale = self.parent.estimate_scale(beta)

        mean = self.parent.family.link.inverse
        varfunc = self.parent.family.variance
        Y = self.parent.Y
        X = self.parent.X
        T = self.parent.T

        # Weights
        VA = (1 - self.a**(2*QX)) / (1 - self.a**2)
        WT = 1 / VA
        WT /= WT.sum()

        QY = []
        for i in range(N):

            if len(Y[i]) == 0:
                continue

            lp = np.dot(X[i], beta)
            E = mean(lp)

            S = np.sqrt(scale*varfunc(E))
            resid = (self.parent.Y[i] - E) / S

            n = len(resid)
            for j1 in range(n):
                for j2 in range(j1):
                    QY.append([resid[j1],resid[j2]])

        QY = np.array(QY)

        # Need to minimize this
        def f(a):
            R = QY[:,0] - (a**QX)*QY[:,1]
            return np.dot(R**2, WT)

        # Left bracket point
        a0,f0 = 0.,f(0.)

        # Center bracket point
        a1,f1 = 0.5,f(0.5)
        while f1 > f0:
            a1 /= 2
            f1 = f(a1)

        # Right bracket point
        a2,f2 = 0.75,f(0.75)
        while f2 < f1:
            a2 = a2 + (1-a2)/2
            f2 = f(a2)
            if a2 > 1 - 1e-6:
                raise ValueError("Autoregressive: unable to find right bracket")

        # Bisection
        while a2 - a0 > 0.001:
            if a2 - a1 > a1 - a0:
                aa = (a1 + a2) / 2
                ff = f(aa)
                if ff > f1:
                    a2,f2 = aa,ff
                else:
                    a0,f0 = a1,f1
                    a1,f1 = aa,ff
            else:
                aa = (a0 + a1) / 2
                ff = f(aa)
                if ff > f1:
                    a0,f0 = aa,ff
                else:
                    a2,f2 = a1,f1
                    a1,f1 = aa,ff

        self.a = a1


    def variance_matrix(self, E, index):
        n = len(E)
        if self.a == 0:
            return np.eye(n, dtype=np.float64),True
        I = np.arange(n)
        return self.a**np.abs(I[:,None] - I[None,:]),True

    def summary(self):

        print "Autoregressive(1) dependence parameter: %.3f\n" % self.a



class GlobalOddsRatio(VarStruct):
    """
    Estimate the global odds ratio for a GEE with either ordinal or nominal data.

    References
    ----------
    PJ Heagerty and S Zeger. "Marginal Regression Models for Clustered Ordinal
    Measurements". Journal of the American Statistical Association Vol. 91,
    Issue 435 (1996).

    Generalized Estimating Equations for Ordinal Data: A Note on Working Correlation Structures
    Thomas Lumley Biometrics Vol. 52, No. 1 (Mar., 1996), pp. 354-361
    http://www.jstor.org/stable/2533173
    """

    def initialize(self, parent, IY, BTW):
        super(GlobalOddsRatio, self).initialize(parent)

        self.IY = IY
        self.BTW = BTW

        # Initialize the dependence parameters
        self.COR = self.observed_crude_oddsratio()
        self.OR = self.COR


    def pooled_odds_ratio(self, A):
        """
        Returns the pooled odds ratio for the list A of 2x2 tables.

        The pooled odds ratio is the inverse variance weighted average of the
        sample odds ratios of the tables.
        """

        if len(A) == 0:
            return 1.

        # Get the samepled odds ratios and variances
        LOR,VA = [],[]
        for B in A:
            lor = np.log(B[1,1]) + np.log(B[0,0]) -\
                  np.log(B[0,1]) - np.log(B[1,0])
            LOR.append(lor)
            VA.append(1/float(B[1,1]) + 1/float(B[1,0]) +\
                      1/float(B[0,1]) + 1/float(B[0,0]))

        # Calculate the inverse variance weighted average
        WT = [1/V for V in VA]
        s = sum(WT)
        WT = [w/s for w in WT]
        por = sum([w*e for w,e in zip(WT,LOR)])

        return np.exp(por)


    def variance_matrix(self, E, index):

        V = self.get_eyy(E, index)
        V -= np.outer(E, E)
        return V,False


    def observed_crude_oddsratio(self):
        """The crude odds ratio is obtained by pooling all data corresponding
        to a given pair of cut points (c,c'), then forming the inverse
        variance weighted average of these odds ratios to obtain a
        single OR.  Since the covariate effects are ignored, this OR
        will generally be greater than the stratified OR.
        """

        BTW = self.BTW
        Y = self.parent.Y

        # Storage for the contingency tables for each (c,c')
        A = {}
        for ii in BTW[0].keys():
            A[ii] = np.zeros((2,2), dtype=np.float64)

        # Get the observed crude OR
        for i in range(len(Y)):

            if len(Y[i]) == 0:
                continue

            # The observed joint values for the current cluster
            y = Y[i]
            Y11 = np.outer(y, y)
            Y10 = np.outer(y, 1-y)
            Y01 = np.outer(1-y, y)
            Y00 = np.outer(1-y, 1-y)

            btw = BTW[i]

            for ky in btw.keys():
                ix = btw[ky]
                A[ky][1,1] += Y11[ix[:,0],ix[:,1]].sum()
                A[ky][1,0] += Y10[ix[:,0],ix[:,1]].sum()
                A[ky][0,1] += Y01[ix[:,0],ix[:,1]].sum()
                A[ky][0,0] += Y00[ix[:,0],ix[:,1]].sum()

        return self.pooled_odds_ratio(A.values())



    def get_eyy(self, EY, index):
        """
        Returns a matrix V such that V[i,j] is the joint probability
        that EY[i] = 1 and EY[j] = 1, based on the marginal
        probabilities in EY and the odds ratio cor.
        """

        cor = self.OR
        IY = self.IY[index]

        # The between-observation joint probabilities
        if cor == 1.0:
            V = np.outer(EY, EY)
        else:
            PS = EY[:,None] + EY[None,:]
            PP = EY[:,None] * EY[None,:]
            S = np.sqrt((1 + PS*(cor-1))**2 + 4*cor*(1-cor)*PP)
            V = 1 +  PS*(cor - 1) - S
            V /= 2*(cor - 1)

        # Fix E[YY'] for elements that belong to same observation
        for iy in IY:
            ey = EY[iy[0]:iy[1]]
            if self.parent.ytype == "ordinal":
                eyr = np.outer(ey, np.ones(len(ey)))
                eyc = np.outer(np.ones(len(ey)), ey)
                V[iy[0]:iy[1],iy[0]:iy[1]] = np.where(eyr < eyc, eyr, eyc)
            else:
                V[iy[0]:iy[1],iy[0]:iy[1]] = np.diag(ey)

        return V


    def update(self, beta):
        """Update the global odds ratio based on the current value of beta."""

        X = self.parent.X
        Y = self.parent.Y
        BTW = self.BTW

        N = len(Y)

        # This will happen if all the clusters have only
        # one observation
        if len(BTW[0]) == 0:
            return

        A = {}
        for ii in BTW[0]:
            A[ii] = np.zeros((2,2), dtype=np.float64)

        for i in range(N):

            if len(Y[i]) == 0:
                continue

            LP = np.dot(X[i], beta)
            ELP = np.exp(-LP)
            EY = 1 / (1 + ELP)

            E11 = self.get_eyy(EY, i)
            E10 = EY[:,None] - E11
            E01 = -E11 + EY
            E00 = 1 - (E11 + E10 + E01)

            btw = BTW[i]

            for ky in btw.keys():
                ix = btw[ky]
                A[ky][1,1] += E11[ix[:,0],ix[:,1]].sum()
                A[ky][1,0] += E10[ix[:,0],ix[:,1]].sum()
                A[ky][0,1] += E01[ix[:,0],ix[:,1]].sum()
                A[ky][0,0] += E00[ix[:,0],ix[:,1]].sum()

        ECOR = self.pooled_odds_ratio(A.values())

        self.OR *= self.COR / ECOR


    def summary(self):

        print "Global odds ratio: %.3f\n" % self.OR
