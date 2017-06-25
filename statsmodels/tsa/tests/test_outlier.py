from numpy.testing import assert_almost_equal
import statsmodels.api as sm

from  statsmodels.tsa.outlier import isoutlier, replace

class TestOutliers:
    data = sm.datasets.nile.load_pandas().data.volume
    data.index = sm.tsa.datetools.dates_from_range('1871', '1970')
    
    def test_isoutlier():
        assert(tsa.isoutlier(data).sum() == 1)
        assert(tsa.isoutlier(data, find='MAD').sum() == 2)
        assert(tsa.isoutlier(data, find='ZScore').sum() == 2)

    def test_replace_NaN():
        assert(tsa.replace(data).isnull().sum() ==1)
        assert(tsa.replace(data, find='MAD').isnull().sum() == 2)
        assert(tsa.replace(data, find='ZScore').isnull().sum() == 2)

    def test_replace_interpolate():
        assert(tsa.replace(data,  how='interpolate').isnull().sum() == 0)
        assert(tsa.replace(data, find='MAD',  how='interpolate').isnull().sum() == 0)
        assert(tsa.replace(data, find='ZScore',  how='interpolate').isnull().sum() == 0)
        
        
if __name__ == '__main__':
    import nose

    nose.runmodule()