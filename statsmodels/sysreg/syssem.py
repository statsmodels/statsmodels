import numpy as np
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

    Notes
    -----
    The larger set of instruments is used for estimation, including
    all of the exogenous variables in the system and the others instruments
    provided in 'instruments' parameters. 
    """

    def __init__(self, sys, instruments=None, dfk=None):
        super(SysSEM, self).__init__(sys, dfk)
        self.instruments = instruments
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

class Sys2SLS(SysSEM):
    pass

