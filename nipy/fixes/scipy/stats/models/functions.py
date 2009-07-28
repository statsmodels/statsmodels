'''
Utility functions for data manipulation
'''

import numpy as np
import numpy.lib.recfunctions as nprf

def xi(data, col=None, time=None, drop=False):
    '''
    Returns an array changing categorical variables to dummy variables.

    Returns an interaction expansion on the specified variable.

    Take a structured or record array and returns an array with categorical
    variables.


    Notes
    -----
    This returns a dummy variable for EVERY distinct string.  If noconsant
    then this is okay.  Otherwise, a "intercept" needs to be designated in
    regression.  Note that STATA returns which variable is omitted when this
    is called. And it is called at runtime of fit...

    Returns the same array as it's given right now (recarray and structured
    array only).

    Where should categoricals home be?
    It is used by every (?) model.

    In STATA, you use xi -- interaction expansion for turning categorical
    into indicator variables, perhaps this is a better name.

    Should also be able to handle numeric data in addition to strings.

    Default drops the "first" group (how to define? -- have an attribute
    "dropped"?)

    Also allows to define dropped as "prevalent" for most prevalent
    or to define which variable -- the latter may be our best option for now.
    '''

#needs error checking
    if isinstance(col, int):
        col = data.dtype.names[col]
    if data.dtype.names and isinstance(col,str):
        tmp_arr = np.unique(data[col])
        tmp_dummy = (tmp_arr[:,np.newaxis]==data[col]).astype(float)
        if drop is True:
            data=nprf.drop_fields(data, col, usemask=False,
            asrecarray=type(data) is np.recarray)
        data=nprf.append_fields(data, tmp_arr, data=tmp_dummy, usemask=False,
                            asrecarray=type(data) is np.recarray)
# TODO: need better column names for numerical indicators
        return data
    else:
# issue a warning?
        return data

def add_constant(data):
    '''
    This appends a constant to the design matrix.

    It checks to make sure a constant is not already included.  If there is
    at least one column of ones then an array of the original design is
    returned.

    Parameters
    ----------
    data : array-like
        `data` is the column-ordered design matrix

    Returns
    -------
    data : array
        The original design matrix with a constant (column of ones)
        as the last column.
    '''
    data = np.asarray(data)
    if np.any(data[0]==1):
        ind = np.squeeze(np.where(data[0]==1))
        if ind.size == 1 and np.all(data[:,ind] == 1):
            return data
        elif ind.size > 1:
            for col in ind:
                if np.all(data[:,col] == 1):
                    return data
    data = np.hstack((data, np.ones((data.shape[0], 1))))
    return data

#class HCCM(OLSModel):
#    """
#    Heteroskedasticity-Corrected Covariance Matrix estimation

#    Parameters
#    ----------
#    design : array-like
#
#    Methods
#    -------
#    Same as OLSModel except that fit accepts an additional argument
#    to define the type of correction.

#    """

#    def __init__(self, design, hascons=True):
#        super(HCCM, self).__init__(design, hascons)

#    def fit(self, Y, robust='HC0'):
#        '''
#        Parameters
#        -----------
#        Y : array-like
#            Response variable

#        robust : string, optional
#            Estimation of the heteroskedasticity robust covariance matrix
#            Values can be "HC0", "HC1", "HC2", or "HC3"
#            Default is HC0

#        '''

#        Z = self.whiten(Y)
#        lfit = RegressionResults(np.dot(self.calc_beta, Z), Y,
#                       normalized_cov_beta=self.normalized_cov_beta)
#        lfit.predict = np.dot(self.design, lfit.beta)
#        lfit.resid = Z - np.dot(self.wdesign, lfit.beta)
#        if robust is "HC0": # HC0 (White 1980)
#            lfit.scale = np.diag(lfit.resid**2)
#        elif robust is "HC1": # HC1-3 MacKinnon and White (1985)
#            lfit.scale = lfit.n/(lfit.n-lfit.df_model-1)*(np.diag(lfit.resid**2))
#        elif robust is "HC2":
#            h=np.diag(np.dot(np.dot(self.wdesign,self.normalized_cov_beta),
#                    self.wdesign.T))
#            lfit.scale=np.diag(lfit.resid**2/(1-h))
#        elif robust is "HC3":
#             h=np.diag(np.dot(np.dot(self.wdesign,self.normalized_cov_beta),
#                    self.wdesign.T))
#             lfit.scale=np.diag((lfit.resid/(1-h))**2)
#        else:
#            raise ValueError, "Robust option %s not understood" % robust
#        lfit.df_resid = self.df_resid
#        lfit.df_model = self.df_model
#        lfit.Z = Z
#        lfit.calc_beta = self.calc_beta # needed for cov_beta()
#        return lfit

