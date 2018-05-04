from statsmodels.compat.python import lrange, lmap, iterkeys, iteritems
import numpy as np
from scipy import stats
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import nottest

def _kurtosis(a):
    '''wrapper for scipy.stats.kurtosis that returns nan instead of raising Error

    missing options
    '''
    try:
        res = stats.kurtosis(a)
    except ValueError:
        res = np.nan
    return res

def _skew(a):
    '''wrapper for scipy.stats.skew that returns nan instead of raising Error

    missing options
    '''
    try:
        res = stats.skew(a)
    except ValueError:
        res = np.nan
    return res

_sign_test_doc = '''
    Signs test.

    Parameters
    ----------
    samp : array-like
        1d array. The sample for which you want to perform the signs
        test.
    mu0 : float
        See Notes for the definition of the sign test. mu0 is 0 by
        default, but it is common to set it to the median.

    Returns
    ---------
    M, p-value

    Notes
    -----
    The signs test returns

    M = (N(+) - N(-))/2

    where N(+) is the number of values above `mu0`, N(-) is the number of
    values below.  Values equal to `mu0` are discarded.

    The p-value for M is calculated using the binomial distrubution
    and can be intrepreted the same as for a t-test. The test-statistic
    is distributed Binom(min(N(+), N(-)), n_trials, .5) where n_trials
    equals N(+) + N(-).

    See Also
    ---------
    scipy.stats.wilcoxon
    '''

@nottest
def sign_test(samp, mu0=0):
    samp = np.asarray(samp)
    pos = np.sum(samp > mu0)
    neg = np.sum(samp < mu0)
    M = (pos-neg)/2.
    p = stats.binom_test(min(pos,neg), pos+neg, .5)
    return M, p
sign_test.__doc__ = _sign_test_doc

class Describe(object):
    '''
    Calculates descriptive statistics for data.

    Defaults to a basic set of statistics, "all" can be specified, or a list
    can be given.

    Parameters
    ----------
    dataset : array-like
        2D dataset for descriptive statistics.
    '''
    def __init__(self, dataset):
        self.dataset = dataset

        #better if this is initially a list to define order, or use an
        # ordered dict. First position is the function
        # Second position is the tuple/list of column names/numbers
        # third is are the results in order of the columns
        self.univariate = dict(
            obs = [len, None, None],
            mean = [np.mean, None, None],
            std = [np.std, None, None],
            min = [np.min, None, None],
            max = [np.max, None, None],
            ptp = [np.ptp, None, None],
            var = [np.var, None, None],
            mode_val = [self._mode_val, None, None],
            mode_bin = [self._mode_bin, None, None],
            median = [np.median, None, None],
            skew = [stats.skew, None, None],
            uss = [lambda x: np.sum(np.asarray(x)**2, axis=0), None, None],
            kurtosis = [stats.kurtosis, None, None],
            percentiles = [self._percentiles, None, None],
            #BUG: not single value
            #sign_test_M = [self.sign_test_m, None, None],
            #sign_test_P = [self.sign_test_p, None, None]
        )

        #TODO: Basic stats for strings
        #self.strings = dict(
            #unique = [np.unique, None, None],
            #number_uniq = [len(
            #most = [
            #least = [

        #TODO: Multivariate
        #self.multivariate = dict(
            #corrcoef(x[, y, rowvar, bias]),
            #cov(m[, y, rowvar, bias]),
            #histogram2d(x, y[, bins, range, normed, weights])
            #)
        self._arraytype = None
        self._columns_list = None

    def _percentiles(self,x):
        p = [stats.scoreatpercentile(x,per) for per in
             (1,5,10,25,50,75,90,95,99)]
        return p
    def _mode_val(self,x):
        return stats.mode(x)[0][0]
    def _mode_bin(self,x):
        return stats.mode(x)[1][0]

    def _array_typer(self):
        """if not a sctructured array"""
        if not(self.dataset.dtype.names):
            """homogeneous dtype array"""
            self._arraytype = 'homog'
        elif self.dataset.dtype.names:
            """structured or rec array"""
            self._arraytype = 'sctruct'
        else:
            assert self._arraytype == 'sctruct' or self._arraytype == 'homog'

    def _is_dtype_like(self, col):
        """
        Check whether self.dataset.[col][0] behaves like a string, numbern
        unknown. `numpy.lib._iotools._is_string_like`
        """
        def string_like():
        #TODO: not sure what the result is if the first item is some type of
        #      missing value
            try:
                self.dataset[col][0] + ''
            except (TypeError, ValueError):
                return False
            return True
        def number_like():
            try:
                self.dataset[col][0] + 1.0
            except (TypeError, ValueError):
                return False
            return True
        if number_like()==True and string_like()==False:
            return 'number'
        elif number_like()==False and string_like()==True:
            return 'string'
        else:
            assert (number_like()==True or string_like()==True), '\
            Not sure of dtype'+str(self.dataset[col][0])

    #@property
    def summary(self, stats='basic', columns='all', orientation='auto'):
        """
        Return a summary of descriptive statistics.

        Parameters
        -----------
        stats: list or str
            The desired statistics, Accepts 'basic' or 'all' or a list.
               'basic' = ('obs', 'mean', 'std', 'min', 'max')
               'all' = ('obs', 'mean', 'std', 'min', 'max', 'ptp', 'var',
                        'mode', 'meadian', 'skew', 'uss', 'kurtosis',
                        'percentiles')
        columns : list or str
          The columns/variables to report the statistics, default is 'all'
          If an object with named columns is given, you may specify the
          column names. For example
        """
        #NOTE
        # standard array: Specifiy column numbers (NEED TO TEST)
        # percentiles currently broken
        # mode requires mode_val and mode_bin separately
        if self._arraytype == None:
            self._array_typer()

        if stats == 'basic':
            stats = ('obs', 'mean', 'std', 'min', 'max')
        elif stats == 'all':
            #stats = self.univariate.keys()
            #dict doesn't keep an order, use full list instead
            stats = ['obs', 'mean', 'std', 'min', 'max', 'ptp', 'var',
                     'mode_val', 'mode_bin', 'median', 'uss', 'skew',
                     'kurtosis', 'percentiles']
        else:
            for astat in stats:
                pass
                #assert astat in self.univariate

        #hack around percentiles multiple output

        #bad naming
        import scipy.stats
        #BUG: the following has all per the same per=99
        ##perdict = dict(('perc_%2d'%per, [lambda x:
         #      scipy.stats.scoreatpercentile(x, per), None, None])
        ##          for per in (1,5,10,25,50,75,90,95,99))

        def _fun(per):
            return lambda x: scipy.stats.scoreatpercentile(x, per)

        perdict = dict(('perc_%02d'%per, [_fun(per), None, None])
                       for per in (1,5,10,25,50,75,90,95,99))

        if 'percentiles' in stats:
            self.univariate.update(perdict)
            idx = stats.index('percentiles')
            stats[idx:idx+1] = sorted(iterkeys(perdict))



        #JP: this doesn't allow a change in sequence, sequence in stats is
        #ignored
        #this is just an if condition
        if any([aitem[1] for aitem in iteritems(self.univariate) if aitem[0] in
                stats]):
            if columns == 'all':
                self._columns_list = []
                if self._arraytype == 'sctruct':
                    self._columns_list = self.dataset.dtype.names
                    #self._columns_list = [col for col in
                    #                      self.dataset.dtype.names if
                            #(self._is_dtype_like(col)=='number')]
                else:
                    self._columns_list = lrange(self.dataset.shape[1])
            else:
                self._columns_list = columns
                if self._arraytype == 'sctruct':
                    for col in self._columns_list:
                        assert (col in self.dataset.dtype.names)
                else:
                    assert self._is_dtype_like(self.dataset) == 'number'

            columstypes = self.dataset.dtype
            #TODO: do we need to make sure they dtype is float64 ?
            for  astat in stats:
                calc = self.univariate[astat]
                if self._arraytype == 'sctruct':
                    calc[1] =  self._columns_list
                    calc[2] = [calc[0](self.dataset[col]) for col in
                            self._columns_list if (self._is_dtype_like(col) ==
                                                      'number')]
                    #calc[2].append([len(np.unique(self.dataset[col])) for col
                                   #in self._columns_list if
                                   #self._is_dtype_like(col)=='string']
                else:
                    calc[1] = ['Col '+str(col) for col in self._columns_list]
                    calc[2] = [calc[0](self.dataset[:,col]) for col in
                               self._columns_list]
            return self.print_summary(stats, orientation=orientation)
        else:
            return self.print_summary(stats, orientation=orientation)

    def print_summary(self, stats, orientation='auto'):
        #TODO: need to specify a table formating for the numbers, using defualt
        title = 'Summary Statistics'
        header = stats
        stubs = self.univariate['obs'][1]
        data = [[self.univariate[astat][2][col] for astat in stats] for col in
                                range(len(self.univariate['obs'][2]))]

        if (orientation == 'varcols') or \
           (orientation == 'auto' and len(stubs) < len(header)):
            #swap rows and columns
            data = lmap(lambda *row: list(row), *data)
            header, stubs = stubs, header

        part_fmt = dict(data_fmts = ["%#8.4g"]*(len(header)-1))
        table = SimpleTable(data,
                            header,
                            stubs,
                            title=title,
                            txt_fmt = part_fmt)

        return table


    def sign_test(self, samp, mu0=0):
        return sign_test(samp, mu0)
    sign_test.__doc__ = _sign_test_doc
    #TODO: There must be a better way but formating the stats of a fuction that
    #      returns 2 values is a problem.
    #def sign_test_m(samp,mu0=0):
        #return self.sign_test(samp,mu0)[0]
    #def sign_test_p(samp,mu0=0):
        #return self.sign_test(samp,mu0)[1]

if __name__ == "__main__":
    #unittest.main()
    t1 = Describe(data4)
    #print(t1.summary(stats='all'))
    noperc = ['obs', 'mean', 'std', 'min', 'max', 'ptp', #'mode',  #'var',
                        'median', 'skew', 'uss', 'kurtosis']
    #TODO: mode var raise exception,
    #TODO: percentile writes list in cell (?), huge wide format
    print(t1.summary(stats=noperc))
    print(t1.summary())
    print(t1.summary( orientation='varcols'))
    print(t1.summary(stats=['mean', 'median', 'min', 'max'], orientation=('varcols')))
    print(t1.summary(stats='all'))


    data1 = np.array([(1,2,'a','aa'),
                      (2,3,'b','bb'),
                      (2,4,'b','cc')],
                     dtype = [('alpha',float), ('beta', int),
                              ('gamma', '|S1'), ('delta', '|S2')])
    data2 = np.array([(1,2),
                      (2,3),
                      (2,4)],
                     dtype = [('alpha',float), ('beta', float)])

    data3 = np.array([[1,2,4,4],
                      [2,3,3,3],
                      [2,4,4,3]], dtype=float)

    data4 = np.array([[1,2,3,4,5,6],
                      [6,5,4,3,2,1],
                      [9,9,9,9,9,9]])

    class TestSimpleTable(object):
        #from statsmodels.iolib.table import SimpleTable, default_txt_fmt

        def test_basic_1(self):
            print('test_basic_1')
            t1 = Describe(data1)
            print(t1.summary())


        def test_basic_2(self):
            print('test_basic_2')
            t2 = Describe(data2)
            print(t2.summary())

        def test_basic_3(self):
            print('test_basic_3')
            t1 = Describe(data3)
            print(t1.summary())

        def test_basic_4(self):
            print('test_basic_4')
            t1 = Describe(data4)
            print(t1.summary())

        def test_basic_1a(self):
            print('test_basic_1a')
            t1 = Describe(data1)
            print(t1.summary(stats='basic', columns=['alpha']))

        def test_basic_1b(self):
            print('test_basic_1b')
            t1 = Describe(data1)
            print(t1.summary(stats='basic', columns='all'))

        def test_basic_2a(self):
            print('test_basic_2a')
            t2 = Describe(data2)
            print(t2.summary(stats='all'))

        def test_basic_3(aself):
            t1 = Describe(data3)
            print(t1.summary(stats='all'))

        def test_basic_4a(self):
            t1 = Describe(data4)
            print(t1.summary(stats='all'))



