from statsmodels.compat.python import iteritems, StringIO
import warnings

from statsmodels.formula.api import ols
from statsmodels.formula.formulatools import make_hypotheses_matrices
from statsmodels.tools import add_constant
from statsmodels.datasets.longley import load, load_pandas

import numpy.testing as npt
from statsmodels.tools.testing import assert_equal
from numpy.testing.utils import WarningManager


longley_formula = 'TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR'

class CheckFormulaOLS(object):

    @classmethod
    def setupClass(cls):
        cls.data = load()

    def test_endog_names(self):
        assert self.model.endog_names == 'TOTEMP'

    def test_exog_names(self):
        assert self.model.exog_names == ['Intercept', 'GNPDEFL', 'GNP',
                                             'UNEMP', 'ARMED', 'POP', 'YEAR']
    def test_design(self):
        npt.assert_equal(self.model.exog,
                         add_constant(self.data.exog, prepend=True))

    def test_endog(self):
        npt.assert_equal(self.model.endog, self.data.endog)

    def test_summary(self):
        # smoke test
        warn_ctx = WarningManager()
        warn_ctx.__enter__()
        try:
            warnings.filterwarnings("ignore",
                                    "kurtosistest only valid for n>=20")
            self.model.fit().summary()
        finally:
            warn_ctx.__exit__()


class TestFormulaPandas(CheckFormulaOLS):
    @classmethod
    def setupClass(cls):
        data = load_pandas().data
        cls.model = ols(longley_formula, data)
        super(TestFormulaPandas, cls).setupClass()


class TestFormulaDict(CheckFormulaOLS):
    @classmethod
    def setupClass(cls):
        data = dict((k, v.tolist()) for k, v in iteritems(load_pandas().data))
        cls.model = ols(longley_formula, data)
        super(TestFormulaDict, cls).setupClass()


class TestFormulaRecArray(CheckFormulaOLS):
    @classmethod
    def setupClass(cls):
        data = load().data
        cls.model = ols(longley_formula, data)
        super(TestFormulaRecArray, cls).setupClass()


def test_tests():
    formula = 'TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR'
    dta = load_pandas().data
    results = ols(formula, dta).fit()
    test_formula = '(GNPDEFL = GNP), (UNEMP = 2), (YEAR/1829 = 1)'
    LC = make_hypotheses_matrices(results, test_formula)
    R = LC.coefs
    Q = LC.constants
    npt.assert_almost_equal(R, [[0, 1, -1, 0, 0, 0, 0],
                               [0, 0 , 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 1./1829]], 8)
    npt.assert_array_equal(Q, [[0],[2],[1]])


def test_formula_labels():
    # make sure labels pass through patsy as expected
    # data(Duncan) from car in R
    dta = StringIO(""""type" "income" "education" "prestige"\n"accountant" "prof" 62 86 82\n"pilot" "prof" 72 76 83\n"architect" "prof" 75 92 90\n"author" "prof" 55 90 76\n"chemist" "prof" 64 86 90\n"minister" "prof" 21 84 87\n"professor" "prof" 64 93 93\n"dentist" "prof" 80 100 90\n"reporter" "wc" 67 87 52\n"engineer" "prof" 72 86 88\n"undertaker" "prof" 42 74 57\n"lawyer" "prof" 76 98 89\n"physician" "prof" 76 97 97\n"welfare.worker" "prof" 41 84 59\n"teacher" "prof" 48 91 73\n"conductor" "wc" 76 34 38\n"contractor" "prof" 53 45 76\n"factory.owner" "prof" 60 56 81\n"store.manager" "prof" 42 44 45\n"banker" "prof" 78 82 92\n"bookkeeper" "wc" 29 72 39\n"mail.carrier" "wc" 48 55 34\n"insurance.agent" "wc" 55 71 41\n"store.clerk" "wc" 29 50 16\n"carpenter" "bc" 21 23 33\n"electrician" "bc" 47 39 53\n"RR.engineer" "bc" 81 28 67\n"machinist" "bc" 36 32 57\n"auto.repairman" "bc" 22 22 26\n"plumber" "bc" 44 25 29\n"gas.stn.attendant" "bc" 15 29 10\n"coal.miner" "bc" 7 7 15\n"streetcar.motorman" "bc" 42 26 19\n"taxi.driver" "bc" 9 19 10\n"truck.driver" "bc" 21 15 13\n"machine.operator" "bc" 21 20 24\n"barber" "bc" 16 26 20\n"bartender" "bc" 16 28 7\n"shoe.shiner" "bc" 9 17 3\n"cook" "bc" 14 22 16\n"soda.clerk" "bc" 12 30 6\n"watchman" "bc" 17 25 11\n"janitor" "bc" 7 20 8\n"policeman" "bc" 34 47 41\n"waiter" "bc" 8 32 10""")
    from pandas import read_table
    dta = read_table(dta, sep=" ")
    model = ols("prestige ~ income + education", dta).fit()
    assert_equal(model.fittedvalues.index, dta.index)


def test_formula_predict():
    from numpy import log
    formula = """TOTEMP ~ log(GNPDEFL) + log(GNP) + UNEMP + ARMED +
                    POP + YEAR"""
    data = load_pandas()
    dta = load_pandas().data
    results = ols(formula, dta).fit()
    npt.assert_almost_equal(results.fittedvalues.values,
                            results.predict(data.exog), 8)


def test_formula_predict_series():
    import pandas as pd
    import pandas.util.testing as tm
    data = pd.DataFrame({"y": [1, 2, 3], "x": [1, 2, 3]}, index=[5, 3, 1])
    results = ols('y ~ x', data).fit()

    result = results.predict(data)
    expected = pd.Series([1., 2., 3.], index=[5, 3, 1])
    tm.assert_series_equal(result, expected)

    result = results.predict(data.x)
    tm.assert_series_equal(result, expected)

    result = results.predict(pd.Series([1, 2, 3], index=[1, 2, 3], name='x'))
    expected = pd.Series([1., 2., 3.], index=[1, 2, 3])
    tm.assert_series_equal(result, expected)

    result = results.predict({"x": [1, 2, 3]})
    expected = pd.Series([1., 2., 3.], index=[0, 1, 2])
    tm.assert_series_equal(result, expected)
