from statsmodels.formula.api import ols
from statsmodels.formula.formulatools import make_hypotheses_matrices
from statsmodels.tools import add_constant
from statsmodels.datasets.longley import load, load_pandas

import numpy.testing as npt

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
        summary = self.model.fit().summary()

class TestFormulaPandas(CheckFormulaOLS):
    @classmethod
    def setupClass(cls):
        data = load_pandas().data
        cls.model = ols(longley_formula, data)
        super(TestFormulaPandas, cls).setupClass()

class TestFormulaDict(CheckFormulaOLS):
    @classmethod
    def setupClass(cls):
        data = dict((k, v.tolist()) for k, v in load_pandas().data.iteritems())
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
