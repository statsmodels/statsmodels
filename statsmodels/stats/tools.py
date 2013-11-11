from collections import OrderedDict

from statsmodels.tools.decorators import nottest

#TODO: Allow options to set digits, control output, etc. of summary

def _check_order(order, keys):
    for i in order:
        if i not in keys:
            raise ValueError(("order is incorrectly specified: {} not in "
                              "keys").format(i))

@nottest
class TestResult:
    def __init__(self, doc, null, alternative, order=None, **kwargs):
        kwargs.update({'null_hypothesis' : 'H0: {}'.format(null)})
        kwargs.update({'alt_hypothesis' : 'HA: {}'.format(alternative)})
        self.__dict__  = kwargs
        self.null_hypothesis = "H0: {}".format(null)
        self.alt_hypothesis = "HA: {}".format(alternative)
        if order is not None:
            _check_order(order, self.__dict__.keys())
        else:
            order = self.__dict__.keys()
        self._order = order
        self.__doc__ = doc

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError("Key not found: {}".format(key))

    def summary(self):
        smry = 'Null Hypothesis: {}\n'.format(self.null_hypothesis)
        smry += 'Alternative Hypothesis: {}\n'.format(self.alt_hypothesis)
        for key in self._order:
            smry += '{}: {}\n'.format(str(key), str(self[key]))
        return smry

if __name__ == "__main__":
    import statsmodels.api as sm

    data = sm.datasets.macrodata.load_pandas()
    x = data.data['realgdp']
    y = data.data['infl']
    res = sm.tsa.adfuller(x, regression='c', autolag=None, maxlag=4)
    d = res[-1]
    res = res[:-1]
    res = dict(zip(
            ['adf', 'pvalue', 'usedlag', 'nobs'], res))

    doc = """
    Attributes
    ----------
    adf : float
        Test statistic
    pvalue : float
        MacKinnon's approximate p-value based on MacKinnon (1994)
    usedlag : int
        Number of lags used.
    nobs : int
        Number of observations used for the ADF regression and calculation of
        the critical values.
    critical values : dict
        Critical values for the test statistic at the 1 %, 5 %, and 10 % levels.
        Based on MacKinnon (2010)
    icbest : float
        The maximized information criterion if autolag is not None.
    regresults : RegressionResults instance
        The
    resstore : (optional) instance of ResultStore
        an instance of a dummy class with results attached as attributes

    """

    res.update(d)
    nice_result = TestResult(doc, 'There is a unit root.',
                             'There is no unit root', ['adf', 'pvalue',
                                                       'usedlag', 'nobs',
                                                       '10%', '5%', '1%'],
                             **res)
