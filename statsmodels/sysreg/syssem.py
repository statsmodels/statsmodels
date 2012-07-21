import numpy as np
import scipy as sp
from statsmodels.sysreg.sysmodel import SysModel, SysResults

def unique_rows(a):
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def unique_cols(a):
    return unique_rows(a.T.copy()).T

class SysSEM(SysModel):
    """
    Two-Stage Least Squares for Simultaneous equations

    Parameters
    ----------
    sys : list of dict
        cf. SysModel. Each equation has now an 'indep_endog' key which is a list
        of the column numbers of the independent endogenous regressors.
    instruments : array
        Array of the exogenous independent variables.
    dkf :
    sigma : 

    Notes
    -----
    The larger set of instruments is used for estimation, including
    all of the exogenous variables in the system and the others instruments
    provided in 'instruments' parameters. 
    """

    def __init__(self, sys, instruments=None, dfk=None, sigma=None):
        super(SysSEM, self).__init__(sys, dfk)
        self.instruments = instruments
        self.sigma = sigma
        self.initialize()

    def initialize(self):
        # Build the instruments design, including all exogs in the system
        # and 'self.instruments' as additionnal instruments.
        exogs = []
        for eq in self.sys:
            id_exog = list(set(range(eq['exog'].shape[1])).difference(
                    eq['indep_endog']))
            exogs.append(eq['exog'][:, id_exog])
        fullexog = np.column_stack(exogs)
        if not(self.instruments is None):
            fullexog = np.hstack((self.instruments, fullexog))
            # Note : the constant is not in the first column. 
            # Does this matter?
        # Delete reoccuring cols
        self.fullexog = unique_cols(fullexog)

    def fit(self):
        z = self.fullexog
        ztzinv = np.linalg.inv(np.dot(z.T, z))
        Pz = np.dot(np.dot(z, ztzinv), z.T)
        
        xhats = [np.dot(Pz, eq['exog']) for eq in self.sys]
        xhat = x = self._compute_sp_exog(xhats)

        omegainv = np.kron(np.linalg.inv(self.sigma), np.identity(self.nobs))
        xtomegainv = x.T * omegainv
        w = np.linalg.inv(xtomegainv * x)
        ww = np.dot(w, xtomegainv)
        params = np.squeeze(np.dot(ww, self.endog.reshape(-1, 1)))
        return params

    def predict(self, params, exog=None):
        '''
        Parameters
        ----------
        exog : None or list of ndarray
            List of individual design (one for each equation)
        '''
        if exog is None:
            sp_exog = self.sp_exog
        else:
            sp_exog = self._compute_sp_exog(exog)

        return sp_exog * params

class Sys2SLS(SysSEM):
    def __init__(self, sys, instruments=None, dfk=None):
        neqs = len(sys)
        super(Sys2SLS, self).__init__(sys, instruments=instruments,
                sigma=np.identity(neqs), dfk=dfk)

