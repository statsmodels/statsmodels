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



class PanelData(object):
    def __init__(self, dataset, endog, exog, time='time', panel=None):
        self.parse_dataset(dataset)
        self.endog_name = endog
        self.exog_name = exog
        self.time_name = time
        self.panel_name = panel

    def parse_dataset(self, dataset)
        if isinstance(dataset, LongPanel):
            pass



#TODO: not sure what the inheritance structure should look like
# Model and LikelihoodModel weren't designed for panels
# maybe should start over?
class PanelModel(PanelData):
    """
    An abstract statistical model class for panel (longitudinal) datasets.

    Parameters
    ---------
    endog : array-like or str
        If a pandas object is used then endog should be the name of the
        endogenous variable as a string.
    exog
    panel_arr
    time_arr
    pandas : pandas.LongPanel object

    Notes
    -----
    If a pandas object is supplied it is assumed that the major_axis is time
    and that the minor_axis has the panel variable.
    """
    def __init__(self, endog=None, exog=None, panel_arr=None, time_arr=None,
            aspandas=False, endog_name=None, exog_name=None, panel_name=None):
        if aspandas == False:
            if endog == None and exog == None and panel_arr == None and \
                    time_arr == None:
                raise ValueError, "If aspandas is False then endog, exog, \
panel_arr, and time_arr cannot be None."
            else:
               self.initialize(endog, exog, panel_arr, time_arr, endog_name,
                        exog_name, panel_name)
        elif aspandas != False:
            if not isinstance(endog, str):
                raise ValueError, "If a pandas object is supplied then endog \
must be a string containing the name of the endogenous variable"
            if not isinstance(aspandas, LongPanel):
                raise ValueError, "Only pandas.LongPanel objects are supported"
            self.initialize_pandas(endog, aspandas, panel_name)


    def initialize(self, endog, exog, panel_arr, time_arr, endog_name,
            exog_name, panel_name):
        """
        Initialize plain array model.

        See PanelModel
        """
                self.endog = np.squeeze(np.asarray(endog))
                self.exog = np.asarray(exog)
                self.panel_arr = np.asanyarray(panel_arr)
                self.time_arr = np.asanyarray(time_arr)
#TODO: add some dimension checks, etc.

    def initialize_pandas(self, endog, aspandas):
        """
        Initialize pandas objects.

        See PanelModel.
        """
        self.aspandas = aspandas
        endog = aspandas[endog].values
        self.endog = np.squeeze(endog)
        exog_name = aspandas.columns.tolist()
        exog_name.remove(endog)
        self.exog = aspandas.filterItems(exog_name).values
#TODO: can the above be simplified to slice notation?
        if panel_name != None:
            self.panel_name = panel_name
        self.exog_name = exog_name
        self.endog_name = endog
        self.time_arr = aspandas.major_axis
        #TODO: is time always handled correctly in fromRecords?
        self.panel_arr = aspandas.minor_axis
#TODO: all of this might need to be refactored to explicitly rely (internally)
# on the pandas LongPanel structure for speed and convenience.
# not sure this part is finished...

#TODO: Use kwd arguments or have fit_method methods?
    def fit(self, method=None, *opts):
        """
        method : LSDV, demeaned, MLE, GLS, BE, FE
        var
        """
        raise NotImplementedError

    def fit_lsdv(self, errors='oneway'):
        """
        Fit using least squares dummy variables.

        Notes
        -----
        Should only be used for small `nobs`.
        """
        pdummies = None
        tdummies = None



#Methods LSDV, GLS, MLE,
class OneWayErrors(PanelModel):
    pass

class TwoWayErrors(PanelModel):
    pass

class SURPanel(PanelModel):
    pass

class SEMPanel
    pass

class DynamicPanel
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




