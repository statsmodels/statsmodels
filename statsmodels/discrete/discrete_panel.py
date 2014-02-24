import numpy as np
from scipy import special

from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.panel.panel_model import PanelModel, _effects_levels
from statsmodels.discrete.discrete_model import Poisson

class PanelPoisson(PanelModel, GenericLikelihoodModel):#, Poisson):
#NOTE: if we want to inherit also from Poisson, we need to add
#      *args, **kwargs to discrete model inits
    """
    References
    ----------
    Cameron, C. and Trivedi, P. K. 1998. *Regression Analysis of Count Data*
        1st Edition.
    """
    # one-way fixed effects only for now using conditional MLE of
    # Hausman et al (1984)
    # panel poisson avoids the incidental parameters problem by conditioning
    # on sufficient statistics for the fixed effects, namely
    # sum(s, y_is) for panel fixed effects and sum(j, y_jt) for time
    # fixed effects. If you want to do two-way just include dummies for
    # the fixed dimension (usually time in econometrics as T = Fixed and
    # N -> inf
    def __init__(self, y, X, effects="oneway", panel=None, time=None,
                 hasconst=None, missing='none'):
        self.effects = effects
        self._effects_level = _effects_levels[effects]
        super(PanelPoisson, self).__init__(y, X, missing=missing,
                                           time=time, panel=panel,
                                           hasconst=hasconst)

    def loglike(self, params, concentrated=True):
        y = self.endog
        offset = getattr(self, "offset", 0)
        exposure = getattr(self, "exposure", 0)
        # lambda in the literature
        XB = np.dot(self.exog, params) + offset + exposure
        mu = np.exp(XB)

        g = self.data.groupings
        level = self._effects_level[0]

        p_it = g.transform_array(mu, lambda x : x / x.sum(), level=level)
        llf = sum(y * np.log(p_it))
        if not concentrated:
            llf += sum(special.gammaln(g.transform_array(y,
                       lambda x : x.sum(), level=level) + 1))
            llf -= sum(g.transform_array(special.gammaln(y + 1),
                         lambda x : x.sum(), level=level))
        return llf

    def score(self, params):
        pass

    def hessian(self, params):
        pass


class PanelZIPoisson(object):
    def __init__(self, y, X, inflateX=None, missing='none'):
        pass

def wide_to_long(df, stubnames, i, j):
    """
    User-friendly wide panel to long format.

    Parameters
    ----------
    df : DataFrame
        The wide-format DataFrame
    stubnames : list
        A list of stub names. The wide format variables are assumed to
        start with the stub names.
    i : str
        The name of the id variable.
    j : str
        The name of the subobservation variable.

    Returns
    -------
    DataFrame
        A DataFrame that contains each stub name as a variable as well as
        variables for i and j.

    Examples
    --------
    import pandas as pd
    import numpy as np
    np.random.seed(123)
    df = pd.DataFrame({"A1970" : {0 : "a", 1 : "b", 2 : "c"},
                       "A1980" : {0 : "d", 1 : "e", 2 : "f"},
                       "B1970" : {0 : 2.5, 1 : 1.2, 2 : .7},
                       "B1980" : {0 : 3.2, 1 : 1.3, 2 : .1},
                       "X"     : dict(zip(range(3), np.random.randn(3)))
                      })
    df["id"] = df.index
    wide_to_long(df, ["A", "B"], i="id", j="year")

    Notes
    -----
    All extra variables are treated as extra id variables.
    """
    def get_var_names(df, regex):
        return df.filter(regex=regex).columns.tolist()

    def melt_stub(df, stub, i, j):
        varnames = get_var_names(df, "^"+stub)
        newdf = pd.melt(df, id_vars=i, value_vars=varnames,
                         value_name=stub, var_name=j)
        newdf[j] = newdf[j].str.replace(stub, "").astype(int)
        return newdf

    id_vars = get_var_names(df, "^(?!%s)" % "|".join(stubnames))
    if i not in id_vars:
        id_vars += [i]

    stub = stubnames.pop(0)
    newdf = melt_stub(df, stub, id_vars, j)

    for stub in stubnames:
        new = melt_stub(df, stub, id_vars, j)
        newdf = newdf.merge(new, how="outer", on=id_vars + [j], copy=False)
    return newdf.set_index([i, j])


#NOTE: can remove this
def read_fwf_file(url, names, sep):
    import pandas as pd
    from urllib2 import urlopen
    import re
    lines = urlopen(url).readlines()

    result = []
    row = []
    def pop_line(lines, sep):
        line = lines.pop(0)
        line = line.strip()
        line = re.split(sep, line)
        return line

    while lines:
        line = pop_line(lines, sep)
        for i in range(len(names)):
            if line:
                row.append(line.pop(0))
            else:
                if lines:
                    line = pop_line(lines, sep)
                    row.append(line.pop(0))
        result.append(row)
        row = []
    return pd.DataFrame(result, columns=names)

if __name__ == "__main__":
    import pandas as pd
    #url = "http://cameron.econ.ucdavis.edu/mmabook/patr7079.asc"
    #names = ["CUSIP", "ARDSSIC", "SCISECT", "LOGK", "SUMPAT", "LOGR70",
    #         "LOGR71", "LOGR72", "LOGR73", "LOGR74", "LOGR75", "LOGR76",
    #         "LOGR77", "LOGR78", "LOGR79", "PAT70", "PAT71", "PAT72", "PAT73",
    #         "PAT74", "PAT75", "PAT76", "PAT77", "PAT78", "PAT79"]
    #dta = read_fwf_file(url, names, sep=" *")

    url = "https://gist.github.com/jseabold/6652233/raw/44a04a78be85f5b3af16f26f51a143e047a10972/patr7079.csv"
    dta = pd.read_csv(url)
    dta["id"] = dta.index

    dta = wide_to_long(dta, ["LOGR", "PAT"], "id", "year")
    dta.reset_index(inplace=True)
    dta = dta.ix[dta["year"] >= 75]
    dta.set_index(["id", "year"], inplace=True)

    y = dta["PAT"]
    dta["const"] = 1
    X = dta[["LOGR"]]
    mod = PanelPoisson(y, X)


    params = [-.0377642474582]
    res = mod.fit(disp=0)
    np.testing.assert_almost_equal(res.params, params, 4)
