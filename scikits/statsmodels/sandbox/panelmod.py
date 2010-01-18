"""
Sandbox Panel Estimators

References
-----------

Baltagi, Badi H. `Econometric Analysis of Panel Data.` 4th ed. Wiley, 2008.
"""

#How to organize the models?
#Right now by error structure?

#Do we have one "panel model" then be able to do any estimations from there
#Is this how the cross-sectional models should be too?
#General Linear Model and all derived from there...probably

from scikits.statsmodels.tools import categorical
import numpy as np

try:
    from pandas import LongPanel, __version__
    __version__ >= .1
except:
    raise ImportError, "While in the sandbox this code depends on the pandas \
package.  http://code.google.com/p/pandas/"

#######

# class PanelModel()

# fit(method=None, **kwargs):




#########

def repanel_cov(groups, sigmas):
    '''calculate error covariance matrix for random effects model

    Parameters
    ----------
    groups : array, (nobs, nre) or (nobs,)
        array of group/category observations
    sigma : array, (nre+1,)
        array of standard deviations of random effects,
        last element is the standard deviation of the
        idiosyncratic error

    Returns
    -------
    omega : array, (nobs, nobs)
        covariance matrix of error
    omegainv : array, (nobs, nobs)
        inverse covariance matrix of error
    omegainvsqrt : array, (nobs, nobs)
        squareroot inverse covariance matrix of error
        such that omega = omegainvsqrt * omegainvsqrt.T

    Notes
    -----
    This does not use sparse matrices and constructs nobs by nobs
    matrices. Also, omegainvsqrt is not sparse, i.e. elements are non-zero
    '''

    if groups.ndim == 1:
        groups = groups[:,None]
    nobs, nre = groups.shape
    omega = sigmas[-1]*np.eye(nobs)
    for igr in range(nre):
        group = groups[:,igr:igr+1]
        groupuniq = np.unique(group)
        dummygr = sigmas[igr] * (group == groupuniq).astype(float)
        omega +=  np.dot(dummygr, dummygr.T)
    ev, evec = np.linalg.eigh(omega)  #eig doesn't work
    omegainv = np.dot(evec, (1/ev * evec).T)
    omegainvhalf = evec/np.sqrt(ev)
    return omega, omegainv, omegainvhalf



#class PanelData(object):
#    def __init__(self, dataset, endog, exog, time='time', panel=None):
#        self.parse_dataset(dataset)
#        self.endog_name = endog
#        self.exog_name = exog
#        self.time_name = time
#        self.panel_name = panel

#    def parse_dataset(self, dataset):
#        if isinstance(dataset, LongPanel):
#            pass

class PanelData(LongPanel):
    pass



#TODO: not sure what the inheritance structure should look like
# Model and LikelihoodModel weren't designed for panels
# maybe should start over?
class PanelModel(object):
    """
    An abstract statistical model class for panel (longitudinal) datasets.

    Parameters
    ---------
    endog : array-like or str
        If a pandas object is used then endog should be the name of the
        endogenous variable as a string.
#    exog
#    panel_arr
#    time_arr
    panel_data : pandas.LongPanel object

    Notes
    -----
    If a pandas object is supplied it is assumed that the major_axis is time
    and that the minor_axis has the panel variable.
    """
#    def __init__(self, endog=None, exog=None, panel_arr=None, time_arr=None,
#            aspandas=False, endog_name=None, exog_name=None, panel_name=None):
#        if aspandas == False:
#            if endog == None and exog == None and panel_arr == None and \
#                    time_arr == None:
#                raise ValueError, "If aspandas is False then endog, exog, \
#panel_arr, and time_arr cannot be None."
#            else:
#               self.initialize(endog, exog, panel_arr, time_arr, endog_name,
#                        exog_name, panel_name)
#        elif aspandas != False:
#            if not isinstance(endog, str):
#                raise ValueError, "If a pandas object is supplied then endog \
#must be a string containing the name of the endogenous variable"
#            if not isinstance(aspandas, LongPanel):
#                raise ValueError, "Only pandas.LongPanel objects are supported"
#            self.initialize_pandas(endog, aspandas, panel_name)


    def __init__(self, panel_data, endog_name, exog_name=None):
        if not isinstance(panel_data, (LongPanel, PanelData)):
            raise ValueError, "Only pandas.LongPanel or PanelData objects are \
supported"
        if not isinstance(endog_name, str):
            raise ValueError, "endog_name must be a string containing the name\
 of the endogenous variable in panel_data"
        if exog_name != None and not isinstance(exog_name, (str, list)):
            raise ValueError, "exog_name should be a string or a list of \
strings of the endogenous variables in panel_data"

        self.initialize_pandas(panel_data, endog_name, exog_name)

#    def initialize(self, endog, exog, panel_arr, time_arr, endog_name,
#            exog_name, panel_name):
#        """
#        Initialize plain array model.
#
#        See PanelModel
#        """
#                self.endog = np.squeeze(np.asarray(endog))
#                self.exog = np.asarray(exog)
#                self.panel_arr = np.asanyarray(panel_arr)
#                self.time_arr = np.asanyarray(time_arr)
#TODO: add some dimension checks, etc.

#    def initialize_pandas(self, endog, aspandas):
#        """
#        Initialize pandas objects.
#
#        See PanelModel.
#        """
#        self.aspandas = aspandas
#        endog = aspandas[endog].values
#        self.endog = np.squeeze(endog)
#        exog_name = aspandas.columns.tolist()
#        exog_name.remove(endog)
#        self.exog = aspandas.filterItems(exog_name).values
#TODO: can the above be simplified to slice notation?
#        if panel_name != None:
#            self.panel_name = panel_name
#        self.exog_name = exog_name
#        self.endog_name = endog
#        self.time_arr = aspandas.major_axis
        #TODO: is time always handled correctly in fromRecords?
#        self.panel_arr = aspandas.minor_axis
#TODO: all of this might need to be refactored to explicitly rely (internally)
# on the pandas LongPanel structure for speed and convenience.
# not sure this part is finished...

    def initialize_pandas(self, panel_data, endog_name, exog_name):
        self.panel_data = panel_data
        endog = panel_data[endog_name].values # does this create a copy?
        self.endog = np.squeeze(endog)
        if exog_name == None:
            exog_name = panel_data.columns.tolist()
            exog_name.remove(endog_name)
        self.exog = panel_data.filterItems(exog_name).values # copy?
        self._exog_name = exog_name
        self._endog_name = endog_name
        self._timeseries = panel_data.major_axis # might not need these
        self._panelseries = panel_data.minor_axis

#TODO: Use kwd arguments or have fit_method methods?
    def fit(self, method=None, effects='oneway', *opts):
        """
        method : LSDV, demeaned, MLE, GLS, BE, FE
        effects : 'invidividual', 'oneway', or 'twoway'
        """
        method = method.lower()
        if method not in ["lsdv", "demeaned", "mle", "gls", "be",
            "fe"]:
            raise ValueError, "%s not a valid method" % method
        if method == "lsdv":
            self.fit_lsdv(opts)

    def fit_lsdv(self, errors='oneway'):
        """
        Fit using least squares dummy variables.

        Notes
        -----
        Should only be used for small `nobs`.
        """
        pdummies = None
        tdummies = None

# does this even make sense?
class OneWayError(PanelModel):
    pass

class TwoWayError(PanelModel):
    pass

class SURPanel(PanelModel):
    pass

class SEMPanel(PanelModel):
    pass

class DynamicPanel(PanelModel):
    pass

if __name__ == "__main__":
    try:
        import pandas
        pandas.version >= .1
    except:
        raise ImportError, "pandas >= .10 not installed"
    from pandas import LongPanel
    import scikits.statsmodels as sm
    import numpy.lib.recfunctions as nprf

    data = sm.datasets.grunfeld.Load()
    # Baltagi doesn't include American Steel
    endog = data.endog[:-20]
    exog = data.exog[:-20]
    arrpanel = nprf.append_fields(exog, 'investment', endog, float,
            usemask=False)
    panel = LongPanel.fromRecords(arrpanel, major_field='year',
            minor_field='firm')
    panel_mod_oneway = OneWayError(panel, endog_name='investment')
#
#""" Covariance matrix, inverse, and negative sqrt for 2way random effects model
#
#Created on Wed Jan 13 08:44:20 2010
#Author: josef-pktd
#
#
#see also Baltagi (3rd edt) 3.3 THE RANDOM EFFECTS MODEL p.35
#for explicit formulas for spectral decomposition
#but this works also for unbalanced panel
#
#I also just saw: 9.4.2 The Random Effects Model p.176 which is
#partially almost the same as I did
#
#this needs to use sparse matrices for larger datasets
#
#"""
#
#import numpy as np
#

    groups = np.array([0,0,0,1,1,2,2,2])
    #groups = np.array([0,0,0,1,1,1,2,2,2])
    nobs = groups.shape[0]
    groupuniq = np.unique(groups)
    periods = np.array([0,1,2,1,2,0,1,2])
    #periods = np.array([0,1,2,0,1,2,0,1,2])
    perioduniq = np.unique(periods)

    dummygr = (groups[:,None] == groupuniq).astype(float)
    dummype = (periods[:,None] == perioduniq).astype(float)

    sigma = 1.
    sigmagr = np.sqrt(2.)
    sigmape = np.sqrt(3.)

    #dummyall = np.c_[sigma*np.ones((nobs,1)), sigmagr*dummygr,
    #                                           sigmape*dummype]
    #exclude constant ?
    dummyall = np.c_[sigmagr*dummygr, sigmape*dummype]
    # omega is the error variance-covariance matrix for the stacked
    # observations
    omega = np.dot(dummyall, dummyall.T) + sigma* np.eye(nobs)
    print omega
    print np.linalg.cholesky(omega)
    ev, evec = np.linalg.eigh(omega)  #eig doesn't work
    omegainv = np.dot(evec, (1/ev * evec).T)
    omegainv2 = np.linalg.inv(omega)
    omegacomp = np.dot(evec, (ev * evec).T)
    print np.max(np.abs(omegacomp - omega))
    #check
    #print np.dot(omegainv,omega)
    print np.max(np.abs(np.dot(omegainv,omega) - np.eye(nobs)))
    omegainvhalf = evec/np.sqrt(ev)  #not sure whether ev shouldn't be column
    print np.max(np.abs(np.dot(omegainvhalf,omegainvhalf.T) - omegainv))

    # now we can use omegainvhalf in GLS (instead of the cholesky)








    sigmas2 = np.array([sigmagr, sigmape, sigma])
    groups2 = np.column_stack((groups, periods))
    omega_, omegainv_, omegainvhalf_ = repanel_cov(groups2, sigmas2)
    print np.max(np.abs(omega_ - omega))
    print np.max(np.abs(omegainv_ - omegainv))
    print np.max(np.abs(omegainvhalf_ - omegainvhalf))

    # notation Baltagi (3rd) section 9.4.1 (Fixed Effects Model)
    Pgr = reduce(np.dot,[dummygr,
            np.linalg.inv(np.dot(dummygr.T, dummygr)),dummygr.T])
    Qgr = np.eye(nobs) - Pgr
    # within group effect: np.dot(Qgr, groups)
    # but this is not memory efficient, compared to groupstats
    print np.max(np.abs(np.dot(Qgr, groups)))
