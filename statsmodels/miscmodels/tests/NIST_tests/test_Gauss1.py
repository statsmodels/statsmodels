
#Output from test_generator.py using NIST data and results
#It is advised that no changes are done in this file.
#Any desired modification should be generalised and added
#in test_generator.py
 
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.nonlinls import NonlinearLS
from numpy.testing import assert_almost_equal, assert_

class TestNonlinearLS(object):
    pass

class funcGauss1(NonlinearLS):

    def expr(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3, b4, b5, b6, b7, b8 = params
        return b1*np.exp(-b2*x) + b3*np.exp(-(x-b4)**2/b5**2) + b6*np.exp(-(x-b7)**2/b8**2)

    def jacobian(self, params, exog=None):
        if exog is None:
            x = self.exog
        else:
            x = exog
        b1, b2, b3, b4, b5, b6, b7, b8 = params
        return np.column_stack([np.exp(-b2*x),-b1*x*np.exp(-b2*x),np.exp(-(x-b4)**2/b5**2),(2*(x-b4)/b5**2)*b3*np.exp(-(x-b4)**2/b5**2),(2*(x-b4)**2/b5**3)*b3*np.exp(-(x-b4)**2/b5**2),np.exp(-(x-b7)**2/b8**2),(2*(x-b7)/b8**2)*b6*np.exp(-(x-b7)**2/b8**2),(2*(x-b7)**2/b8**3)*b6*np.exp(-(x-b7)**2/b8**2)])

class TestGauss1(TestNonlinearLS):

    def setup(self):
        x = np.array([1.0,
                   2.0,
                   3.0,
                   4.0,
                   5.0,
                   6.0,
                   7.0,
                   8.0,
                   9.0,
                   10.0,
                   11.0,
                   12.0,
                   13.0,
                   14.0,
                   15.0,
                   16.0,
                   17.0,
                   18.0,
                   19.0,
                   20.0,
                   21.0,
                   22.0,
                   23.0,
                   24.0,
                   25.0,
                   26.0,
                   27.0,
                   28.0,
                   29.0,
                   30.0,
                   31.0,
                   32.0,
                   33.0,
                   34.0,
                   35.0,
                   36.0,
                   37.0,
                   38.0,
                   39.0,
                   40.0,
                   41.0,
                   42.0,
                   43.0,
                   44.0,
                   45.0,
                   46.0,
                   47.0,
                   48.0,
                   49.0,
                   50.0,
                   51.0,
                   52.0,
                   53.0,
                   54.0,
                   55.0,
                   56.0,
                   57.0,
                   58.0,
                   59.0,
                   60.0,
                   61.0,
                   62.0,
                   63.0,
                   64.0,
                   65.0,
                   66.0,
                   67.0,
                   68.0,
                   69.0,
                   70.0,
                   71.0,
                   72.0,
                   73.0,
                   74.0,
                   75.0,
                   76.0,
                   77.0,
                   78.0,
                   79.0,
                   80.0,
                   81.0,
                   82.0,
                   83.0,
                   84.0,
                   85.0,
                   86.0,
                   87.0,
                   88.0,
                   89.0,
                   90.0,
                   91.0,
                   92.0,
                   93.0,
                   94.0,
                   95.0,
                   96.0,
                   97.0,
                   98.0,
                   99.0,
                   100.0,
                   101.0,
                   102.0,
                   103.0,
                   104.0,
                   105.0,
                   106.0,
                   107.0,
                   108.0,
                   109.0,
                   110.0,
                   111.0,
                   112.0,
                   113.0,
                   114.0,
                   115.0,
                   116.0,
                   117.0,
                   118.0,
                   119.0,
                   120.0,
                   121.0,
                   122.0,
                   123.0,
                   124.0,
                   125.0,
                   126.0,
                   127.0,
                   128.0,
                   129.0,
                   130.0,
                   131.0,
                   132.0,
                   133.0,
                   134.0,
                   135.0,
                   136.0,
                   137.0,
                   138.0,
                   139.0,
                   140.0,
                   141.0,
                   142.0,
                   143.0,
                   144.0,
                   145.0,
                   146.0,
                   147.0,
                   148.0,
                   149.0,
                   150.0,
                   151.0,
                   152.0,
                   153.0,
                   154.0,
                   155.0,
                   156.0,
                   157.0,
                   158.0,
                   159.0,
                   160.0,
                   161.0,
                   162.0,
                   163.0,
                   164.0,
                   165.0,
                   166.0,
                   167.0,
                   168.0,
                   169.0,
                   170.0,
                   171.0,
                   172.0,
                   173.0,
                   174.0,
                   175.0,
                   176.0,
                   177.0,
                   178.0,
                   179.0,
                   180.0,
                   181.0,
                   182.0,
                   183.0,
                   184.0,
                   185.0,
                   186.0,
                   187.0,
                   188.0,
                   189.0,
                   190.0,
                   191.0,
                   192.0,
                   193.0,
                   194.0,
                   195.0,
                   196.0,
                   197.0,
                   198.0,
                   199.0,
                   200.0,
                   201.0,
                   202.0,
                   203.0,
                   204.0,
                   205.0,
                   206.0,
                   207.0,
                   208.0,
                   209.0,
                   210.0,
                   211.0,
                   212.0,
                   213.0,
                   214.0,
                   215.0,
                   216.0,
                   217.0,
                   218.0,
                   219.0,
                   220.0,
                   221.0,
                   222.0,
                   223.0,
                   224.0,
                   225.0,
                   226.0,
                   227.0,
                   228.0,
                   229.0,
                   230.0,
                   231.0,
                   232.0,
                   233.0,
                   234.0,
                   235.0,
                   236.0,
                   237.0,
                   238.0,
                   239.0,
                   240.0,
                   241.0,
                   242.0,
                   243.0,
                   244.0,
                   245.0,
                   246.0,
                   247.0,
                   248.0,
                   249.0,
                   250.0])
        y = np.array([97.62227,
                   97.807239999999993,
                   96.622470000000007,
                   92.590220000000002,
                   91.238690000000005,
                   95.327039999999997,
                   90.350399999999993,
                   89.462350000000001,
                   91.725200000000001,
                   89.869159999999994,
                   86.880759999999995,
                   85.943600000000004,
                   87.606859999999998,
                   86.258390000000006,
                   80.749759999999995,
                   83.035510000000002,
                   88.258369999999999,
                   82.013159999999999,
                   82.740979999999993,
                   83.300340000000006,
                   81.278499999999994,
                   81.855059999999995,
                   80.751949999999994,
                   80.095730000000003,
                   81.076329999999999,
                   78.815420000000003,
                   78.385959999999997,
                   79.933859999999996,
                   79.484740000000002,
                   79.959419999999994,
                   76.106909999999999,
                   78.398300000000006,
                   81.430599999999998,
                   82.488669999999999,
                   81.654619999999994,
                   80.843230000000005,
                   88.686629999999994,
                   84.744380000000007,
                   86.839340000000007,
                   85.97739,
                   91.285089999999997,
                   97.224109999999996,
                   93.517330000000001,
                   94.101590000000002,
                   101.91759999999999,
                   98.431340000000006,
                   110.42140000000001,
                   107.6628,
                   111.72880000000001,
                   116.5115,
                   120.76090000000001,
                   123.95529999999999,
                   124.2437,
                   130.7996,
                   133.29599999999999,
                   130.77879999999999,
                   132.0565,
                   138.6584,
                   142.92519999999999,
                   142.72149999999999,
                   144.1249,
                   147.43770000000001,
                   148.2647,
                   152.05189999999999,
                   147.38630000000001,
                   149.20740000000001,
                   148.9537,
                   144.58760000000001,
                   148.12260000000001,
                   148.01439999999999,
                   143.88929999999999,
                   140.90880000000001,
                   143.4434,
                   139.3938,
                   135.98779999999999,
                   136.39269999999999,
                   126.72620000000001,
                   124.4487,
                   122.8647,
                   113.8557,
                   113.7037,
                   106.8407,
                   107.0034,
                   102.4629,
                   96.092960000000005,
                   94.575550000000007,
                   86.988240000000005,
                   84.901539999999997,
                   81.180229999999995,
                   76.401169999999993,
                   67.091999999999999,
                   72.671549999999996,
                   68.10848,
                   67.990880000000004,
                   63.340940000000003,
                   60.552529999999997,
                   56.186869999999999,
                   53.644820000000003,
                   53.703069999999997,
                   48.07893,
                   42.212580000000003,
                   45.651809999999998,
                   41.697279999999999,
                   41.249459999999999,
                   39.21349,
                   37.71696,
                   36.683950000000003,
                   37.303930000000001,
                   37.432769999999998,
                   37.450119999999998,
                   32.646479999999997,
                   31.84347,
                   31.399509999999999,
                   26.689119999999999,
                   32.253230000000002,
                   27.61008,
                   33.586489999999998,
                   28.107140000000001,
                   30.264279999999999,
                   28.016480000000001,
                   29.110209999999999,
                   23.020990000000001,
                   25.65091,
                   28.502949999999998,
                   25.237010000000001,
                   26.138280000000002,
                   33.532600000000002,
                   29.251950000000001,
                   27.098469999999999,
                   26.529990000000002,
                   25.524010000000001,
                   26.69218,
                   24.552689999999998,
                   27.71763,
                   25.202970000000001,
                   25.614830000000001,
                   25.068930000000002,
                   27.639299999999999,
                   24.948509999999999,
                   25.86806,
                   22.481829999999999,
                   26.900449999999999,
                   25.399190000000001,
                   17.906140000000001,
                   23.760390000000001,
                   25.896889999999999,
                   27.642309999999998,
                   22.86101,
                   26.470030000000001,
                   23.72888,
                   27.543340000000001,
                   30.52683,
                   28.072610000000001,
                   34.928150000000002,
                   28.29194,
                   34.191609999999997,
                   35.41207,
                   37.093359999999997,
                   40.9833,
                   39.539230000000003,
                   47.801229999999997,
                   47.463050000000003,
                   51.04166,
                   54.580649999999999,
                   57.530009999999997,
                   61.42089,
                   62.790320000000001,
                   68.51455,
                   70.230530000000002,
                   74.427760000000006,
                   76.599109999999996,
                   81.620530000000002,
                   83.422079999999994,
                   79.174509999999998,
                   88.569850000000002,
                   85.66525,
                   86.555019999999999,
                   90.65907,
                   84.272900000000007,
                   85.722200000000001,
                   83.107020000000006,
                   82.168840000000003,
                   80.42568,
                   78.15692,
                   79.796909999999997,
                   77.843779999999995,
                   74.503270000000001,
                   71.572890000000001,
                   65.880309999999994,
                   65.013850000000005,
                   60.195819999999998,
                   59.667259999999999,
                   52.95478,
                   53.877920000000003,
                   44.912739999999999,
                   41.099089999999997,
                   41.68018,
                   34.533790000000003,
                   34.864190000000001,
                   33.147869999999998,
                   29.588640000000002,
                   27.294619999999998,
                   21.914390000000001,
                   19.081589999999998,
                   24.902899999999999,
                   19.823409999999999,
                   16.755510000000001,
                   18.24558,
                   17.235489999999999,
                   16.349340000000002,
                   13.71285,
                   14.75676,
                   13.971690000000001,
                   12.42867,
                   14.35519,
                   7.703309,
                   10.23441,
                   11.783149999999999,
                   13.87768,
                   4.5357000000000003,
                   10.059279999999999,
                   8.4248239999999992,
                   10.53312,
                   9.6022549999999995,
                   7.8775139999999997,
                   6.258121,
                   8.8998650000000001,
                   7.8777540000000004,
                   12.51191,
                   10.662050000000001,
                   6.0354000000000001,
                   6.7906550000000001,
                   8.7835350000000005,
                   4.6002879999999999,
                   8.4009149999999995,
                   7.2165609999999996,
                   10.01741,
                   7.3312780000000002,
                   6.527863,
                   2.8420010000000002,
                   10.32507,
                   4.7909949999999997,
                   8.3771009999999997,
                   6.2644450000000003,
                   2.706213,
                   8.3623290000000008,
                   8.9836580000000001,
                   3.362571,
                   1.1827460000000001,
                   4.8753590000000004])
        mod1 = funcGauss1(y, x)
        self.res_start1 = mod1.fit(start_value=[97.0, 0.0089999999999999993, 100.0, 65.0, 20.0, 70.0, 178.0, 16.5])
        mod2 = funcGauss1(y, x)
        self.res_start2 = mod2.fit(start_value=[94.0, 0.010500000000000001, 99.0, 63.0, 25.0, 71.0, 180.0, 20.0])

    def test_basic(self):
        res1 = self.res_start1
        res2 = self.res_start2
        assert_almost_equal(res1.params,[98.778210870999999, 0.010497276517, 100.48990633, 67.481111275999993, 23.129773360000002, 71.994503003999995, 178.99805021, 18.389389025],decimal=3)
        assert_almost_equal(res2.params,[98.778210870999999, 0.010497276517, 100.48990633, 67.481111275999993, 23.129773360000002, 71.994503003999995, 178.99805021, 18.389389025],decimal=3)
