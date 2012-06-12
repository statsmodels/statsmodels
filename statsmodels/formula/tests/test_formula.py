from statsmodels.formula.api import ols
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
