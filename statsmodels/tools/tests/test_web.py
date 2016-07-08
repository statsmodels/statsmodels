from statsmodels.tools.web import _generate_url, webdoc
from statsmodels.regression.linear_model import OLS
from unittest import TestCase
from nose.tools import assert_equal, assert_raises
from numpy import array

class TestWeb(TestCase):
    def test_string(self):
        url = _generate_url('arch',True)
        assert_equal(url, 'http://www.statsmodels.org/stable/search.html?q=arch&check_keywords=yes&area=default')
        url = _generate_url('arch',False)
        assert_equal(url, 'http://www.statsmodels.org/devel/search.html?q=arch&check_keywords=yes&area=default')
        url = _generate_url('dickey fuller',False)
        assert_equal(url, 'http://www.statsmodels.org/devel/search.html?q=dickey+fuller&check_keywords=yes&area=default')

    def test_function(self):
        url = _generate_url(OLS, True)
        assert_equal(url, 'http://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html')
        url = _generate_url(OLS, False)
        assert_equal(url, 'http://www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.OLS.html')

    def test_nothing(self):
        url = _generate_url(None, True)
        assert_equal(url, 'http://www.statsmodels.org/stable/')
        url = _generate_url(None, False)
        assert_equal(url, 'http://www.statsmodels.org/devel/')

    def test_errors(self):
        assert_raises(ValueError, webdoc, array, True)
        assert_raises(ValueError, webdoc, 1, False)



