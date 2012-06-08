import numpy as np

class Bunch(dict):
    def __init__(self, **kw):
        dict.__init__(self, kw)
        self.__dict__  = self


sur = Bunch()
sur.call = '''systemfit(formula = formula, method = "SUR", data = panel, methodResidCov = "noDfCor")'''
sur.params = np.array([
     0.9979992, 0.06886083, 0.3083878, -21.1374, 0.03705313, 0.1286866, 
     -168.1134, 0.1219063, 0.3821666, 62.25631, 0.1214024, 0.3691114, 
     1.407487, 0.05635611, 0.04290209
    ]).reshape(15,1, order='F')
sur.cov_params = np.array([
     133.7852, -0.1840371, 0.01059675, -31.21063, 0.01337293, 0.01044742, 
     -161.6206, 0.0343593, -0.002197769, 158.6378, -0.06916984, 
     -0.01307428, 0.03036482, 6.95277e-05, 0.006869873, -0.1840371, 
     0.0002886686, -0.0001325481, 0.03921905, -1.927362e-05, -4.504895e-06, 
     0.199845, -5.161935e-05, 3.680446e-05, -0.1667078, 0.0001043746, 
     -0.0001326094, 0.002294617, -1.738464e-06, -1.317451e-05, 0.01059675, 
     -0.0001325481, 0.0006704354, 0.02437061, -1.011555e-07, -6.041142e-05, 
     0.07380307, 1.174274e-05, -0.0001923003, -0.2035738, -2.6258e-05, 
     0.0008660189, -0.00788213, 9.366098e-06, 1.866325e-05, -31.21063, 
     0.03921905, 0.02437061, 635.1519, -0.2770547, -0.160579, 582.7983, 
     -0.1213974, -0.04056555, 716.5924, -0.2598967, -0.5263574, 100.0872, 
     -0.1653628, 0.2297848, 0.01337293, -1.927362e-05, -1.011555e-07, 
     -0.2770547, 0.0001458083, -1.501036e-05, -0.2645839, 6.420246e-05, 
     -2.106559e-05, -0.3380538, 0.0001551097, 0.000109222, -0.038997, 
     8.497461e-05, -0.0002103376, 0.01044742, -4.504895e-06, -6.041142e-05, 
     -0.160579, -1.501036e-05, 0.0004741078, -0.09690712, -8.097892e-06, 
     0.0002035703, -0.02848283, -0.0001030127, 0.0007854907, -0.03888171, 
     9.981992e-07, 0.0004461934, -161.6206, 0.199845, 0.07380307, 582.7983, 
     -0.2645839, -0.09690712, 8026.788, -1.849853, 0.5369888, -1593.177, 
     0.7497236, 0.05596429, 108.7601, -0.2004693, 0.3741887, 0.0343593, 
     -5.161935e-05, 1.174274e-05, -0.1213974, 6.420246e-05, -8.097892e-06, 
     -1.849853, 0.0004695548, -0.0002854946, 0.3432181, -0.0001922897, 
     0.0001219025, -0.0226266, 4.929475e-05, -0.0001219727, -0.002197769, 
     3.680446e-05, -0.0001923003, -0.04056555, -2.106559e-05, 0.0002035703, 
     0.5369888, -0.0002854946, 0.001079986, 0.01136712, 0.0001289726, 
     -0.0009010479, -0.006771958, -2.030504e-05, 0.0002381459, 158.6378, 
     -0.1667078, -0.2035738, 716.5924, -0.3380538, -0.02848283, -1593.177, 
     0.3432181, 0.01136712, 11369.52, -5.187909, -2.525447, 256.2028, 
     -0.4783715, 1.054605, -0.06916984, 0.0001043746, -2.6258e-05, 
     -0.2598967, 0.0001551097, -0.0001030127, 0.7497236, -0.0001922897, 
     0.0001289726, -5.187909, 0.002739435, -0.0007250238, -0.08272505, 
     0.0002158356, -0.0007249087, -0.01307428, -0.0001326094, 0.0008660189, 
     -0.5263574, 0.000109222, 0.0007854907, 0.05596429, 0.0001219025, 
     -0.0009010479, -2.525447, -0.0007250238, 0.0134136, -0.2289551, 
     0.0001790085, 0.001271094, 0.03036482, 0.002294617, -0.00788213, 
     100.0872, -0.038997, -0.03888171, 108.7601, -0.0226266, -0.006771958, 
     256.2028, -0.08272505, -0.2289551, 39.2104, -0.06063476, 0.0689298, 
     6.95277e-05, -1.738464e-06, 9.366098e-06, -0.1653628, 8.497461e-05, 
     9.981992e-07, -0.2004693, 4.929475e-05, -2.030504e-05, -0.4783715, 
     0.0002158356, 0.0001790085, -0.06063476, 0.0001316823, -0.0003235898, 
     0.006869873, -1.317451e-05, 1.866325e-05, 0.2297848, -0.0002103376, 
     0.0004461934, 0.3741887, -0.0001219727, 0.0002381459, 1.054605, 
     -0.0007249087, 0.001271094, 0.0689298, -0.0003235898, 0.001730147
    ]).reshape(15,15, order='F')
sur.cov_params_rownames = ['Chrysler_(Intercept)', 'Chrysler_value', 'Chrysler_capital', 'General.Electric_(Intercept)', 'General.Electric_value', 'General.Electric_capital', 'General.Motors_(Intercept)', 'General.Motors_value', 'General.Motors_capital', 'US.Steel_(Intercept)', 'US.Steel_value', 'US.Steel_capital', 'Westinghouse_(Intercept)', 'Westinghouse_value', 'Westinghouse_capital', ]
sur.cov_params_colnames = ['Chrysler_(Intercept)', 'Chrysler_value', 'Chrysler_capital', 'General.Electric_(Intercept)', 'General.Electric_value', 'General.Electric_capital', 'General.Motors_(Intercept)', 'General.Motors_value', 'General.Motors_capital', 'US.Steel_(Intercept)', 'US.Steel_value', 'US.Steel_capital', 'Westinghouse_(Intercept)', 'Westinghouse_value', 'Westinghouse_capital', ]
sur.resid_cov_est = np.array([
     149.8722, -21.37565, -282.7564, 367.8402, 13.30695, -21.37565, 
     660.8294, 607.5331, 978.4503, 176.4491, -282.7564, 607.5331, 7160.294, 
     -1967.046, 126.1762, 367.8402, 978.4503, -1967.046, 7904.663, 
     511.4995, 13.30695, 176.4491, 126.1762, 511.4995, 88.6617
    ]).reshape(5,5, order='F')
sur.resid_cov_est_rownames = ['Chrysler', 'General.Electric', 'General.Motors', 'US.Steel', 'Westinghouse', ]
sur.resid_cov_est_colnames = ['Chrysler', 'General.Electric', 'General.Motors', 'US.Steel', 'Westinghouse', ]
sur.resid_cov = np.array([
     153.2369, 3.147771, -315.6107, 414.5298, 16.64749, 3.147771, 704.729, 
     601.6316, 1298.695, 201.4385, -315.6107, 601.6316, 7222.22, -2446.317, 
     129.7644, 414.5298, 1298.695, -2446.317, 8174.28, 613.9925, 16.64749, 
     201.4385, 129.7644, 613.9925, 94.90675
    ]).reshape(5,5, order='F')
sur.resid_cov_rownames = ['Chrysler', 'General.Electric', 'General.Motors', 'US.Steel', 'Westinghouse', ]
sur.resid_cov_colnames = ['Chrysler', 'General.Electric', 'General.Motors', 'US.Steel', 'Westinghouse', ]
sur.method = 'SUR'
sur.rank = 15
sur.df_resid = 85
sur.iter = 1
sur.panelLike = '''TRUE'''


sur.equ1 = Bunch()
sur.equ1.eqnLabel = 'Chrysler'
sur.equ1.method = 'SUR'
sur.equ1.residuals = np.array([
     7.304531, 10.92484, -6.305147, 4.473352, -15.22205, -2.397737, 
     -0.1575964, -4.511807, -14.80854, -8.177869, 12.66022, -14.5771, 
     -8.040364, 6.546927, -8.122915, 1.578119, 41.15666, 4.322671, 
     -1.765152, -4.881048
    ]).reshape(20,1, order='F')
sur.equ1.params = np.array([
     0.9979992, 0.06886083, 0.3083878
    ]).reshape(3,1, order='F')
sur.equ1.cov_params = np.array([
     133.7852, -0.1840371, 0.01059675, -0.1840371, 0.0002886686, 
     -0.0001325481, 0.01059675, -0.0001325481, 0.0006704354
    ]).reshape(3,3, order='F')
sur.equ1.cov_params_rownames = ['(Intercept)', 'value', 'capital', ]
sur.equ1.cov_params_colnames = ['(Intercept)', 'value', 'capital', ]
sur.equ1.fittedvalues = np.array([
     32.98547, 61.83516, 72.56515, 47.12665, 67.63205, 71.80774, 68.5076, 
     51.31181, 62.20854, 67.74787, 76.11978, 88.6971, 70.72036, 82.81307, 
     87.10292, 99.08188, 119.4633, 140.6773, 176.6952, 177.371
    ]).reshape(20,1, order='F')
sur.equ1.terms = '''Chrysler_invest ~ Chrysler_value + Chrysler_capital'''
sur.equ1.rank = 3
sur.equ1.nCoef_sys = 15
sur.equ1.rank_sys = 15
sur.equ1.df_resid = 17
sur.equ1.df_resid_sys = 85
sur.equ1.model = '''structure(list(Chrysler_invest = c(40.29, 72.76, 66.26, 51.6,  52.41, 69.41, 68.35, 46.8, 47.4, 59.57, 88.78, 74.12, 62.68,  89.36, 78.98, 100.66, 160.62, 145, 174.93, 172.49), Chrysler_value = c(417.5,  837.8, 883.9, 437.9, 679.7, 727.8, 643.6, 410.9, 588.4, 698.4,  846.4, 893.8, 579, 694.6, 590.3, 693.5, 809, 727, 1001.5, 703.2 ), Chrysler_capital = c(10.5, 10.2, 34.7, 51.8, 64.3, 67.1, 75.2,  71.4, 67.1, 60.5, 54.6, 84.8, 96.8, 110.2, 147.4, 163.2, 203.5,  290.6, 346.1, 414.9)), .Names = c("Chrysler_invest", "Chrysler_value",  "Chrysler_capital"), class = "data.frame", row.names = c("X1935",  "X1936", "X1937", "X1938", "X1939", "X1940", "X1941", "X1942",  "X1943", "X1944", "X1945", "X1946", "X1947", "X1948", "X1949",  "X1950", "X1951", "X1952", "X1953", "X1954"), terms = Chrysler_invest ~      Chrysler_value + Chrysler_capital)'''


sur.equ2 = Bunch()
sur.equ2.eqnLabel = 'General.Electric'
sur.equ2.method = 'SUR'
sur.equ2.residuals = np.array([
     -1.722547, -21.98919, -20.71866, -29.94072, -36.57318, -7.480208, 
     37.75138, 17.16102, -23.55019, -25.92565, -0.7824099, 54.68741, 
     48.21891, 38.06106, -13.1822, -28.34837, 2.535621, 7.938665, 9.774311, 
     -5.915052
    ]).reshape(20,1, order='F')
sur.equ2.params = np.array([
     -21.1374, 0.03705313, 0.1286866
    ]).reshape(3,1, order='F')
sur.equ2.cov_params = np.array([
     635.1519, -0.2770547, -0.160579, -0.2770547, 0.0001458083, 
     -1.501036e-05, -0.160579, -1.501036e-05, 0.0004741078
    ]).reshape(3,3, order='F')
sur.equ2.cov_params_rownames = ['(Intercept)', 'value', 'capital', ]
sur.equ2.cov_params_colnames = ['(Intercept)', 'value', 'capital', ]
sur.equ2.fittedvalues = np.array([
     34.82255, 66.98919, 97.91866, 74.54072, 84.67318, 81.88021, 75.24862, 
     74.73898, 84.85019, 82.72565, 94.38241, 105.2126, 98.98109, 108.2389, 
     111.4822, 121.8484, 132.6644, 149.3613, 169.7257, 195.5151
    ]).reshape(20,1, order='F')
sur.equ2.terms = '''General.Electric_invest ~ General.Electric_value + General.Electric_capital'''
sur.equ2.rank = 3
sur.equ2.nCoef_sys = 15
sur.equ2.rank_sys = 15
sur.equ2.df_resid = 17
sur.equ2.df_resid_sys = 85
sur.equ2.model = '''structure(list(General.Electric_invest = c(33.1, 45, 77.2, 44.6,  48.1, 74.4, 113, 91.9, 61.3, 56.8, 93.6, 159.9, 147.2, 146.3,  98.3, 93.5, 135.2, 157.3, 179.5, 189.6), General.Electric_value = c(1170.6,  2015.8, 2803.3, 2039.7, 2256.2, 2132.2, 1834.1, 1588, 1749.4,  1687.2, 2007.7, 2208.3, 1656.7, 1604.4, 1431.8, 1610.5, 1819.4,  2079.7, 2371.6, 2759.9), General.Electric_capital = c(97.8, 104.4,  118, 156.2, 172.6, 186.6, 220.9, 287.8, 319.9, 321.3, 319.6,  346, 456.4, 543.4, 618.3, 647.4, 671.3, 726.1, 800.3, 888.9)), .Names = c("General.Electric_invest",  "General.Electric_value", "General.Electric_capital"), class = "data.frame", row.names = c("X1935",  "X1936", "X1937", "X1938", "X1939", "X1940", "X1941", "X1942",  "X1943", "X1944", "X1945", "X1946", "X1947", "X1948", "X1949",  "X1950", "X1951", "X1952", "X1953", "X1954"), terms = General.Electric_invest ~      General.Electric_value + General.Electric_capital)'''


sur.equ3 = Bunch()
sur.equ3.eqnLabel = 'General.Motors'
sur.equ3.method = 'SUR'
sur.equ3.residuals = np.array([
     109.3547, -28.47935, -137.9702, 5.477267, -104.6257, -15.99238, 
     27.76434, 104.573, 72.61146, 104.7042, 37.90284, 105.0552, 16.09081, 
     -51.96565, -117.7126, -66.81917, -126.7026, -87.7525, 32.38584, 
     122.1005
    ]).reshape(20,1, order='F')
sur.equ3.params = np.array([
     -168.1134, 0.1219063, 0.3821666
    ]).reshape(3,1, order='F')
sur.equ3.cov_params = np.array([
     8026.788, -1.849853, 0.5369888, -1.849853, 0.0004695548, 
     -0.0002854946, 0.5369888, -0.0002854946, 0.001079986
    ]).reshape(3,3, order='F')
sur.equ3.cov_params_rownames = ['(Intercept)', 'value', 'capital', ]
sur.equ3.cov_params_colnames = ['(Intercept)', 'value', 'capital', ]
sur.equ3.fittedvalues = np.array([
     208.2453, 420.2794, 548.5702, 252.2227, 435.4257, 477.1924, 484.2357, 
     343.427, 426.9885, 442.7958, 523.2972, 583.0448, 552.8092, 581.1657, 
     672.8126, 709.7192, 882.6026, 978.9525, 1272.014, 1364.599
    ]).reshape(20,1, order='F')
sur.equ3.terms = '''General.Motors_invest ~ General.Motors_value + General.Motors_capital'''
sur.equ3.rank = 3
sur.equ3.nCoef_sys = 15
sur.equ3.rank_sys = 15
sur.equ3.df_resid = 17
sur.equ3.df_resid_sys = 85
sur.equ3.model = '''structure(list(General.Motors_invest = c(317.6, 391.8, 410.6,  257.7, 330.8, 461.2, 512, 448, 499.6, 547.5, 561.2, 688.1, 568.9,  529.2, 555.1, 642.9, 755.9, 891.2, 1304.4, 1486.7), General.Motors_value = c(3078.5,  4661.7, 5387.1, 2792.2, 4313.2, 4643.9, 4551.2, 3244.1, 4053.7,  4379.3, 4840.9, 4900.9, 3526.5, 3254.7, 3700.2, 3755.6, 4833,  4924.9, 6241.7, 5593.6), General.Motors_capital = c(2.8, 52.6,  156.9, 209.2, 203.4, 207.2, 255.2, 303.7, 264.1, 201.6, 265,  402.2, 761.5, 922.4, 1020.1, 1099, 1207.7, 1430.5, 1777.3, 2226.3 )), .Names = c("General.Motors_invest", "General.Motors_value",  "General.Motors_capital"), class = "data.frame", row.names = c("X1935",  "X1936", "X1937", "X1938", "X1939", "X1940", "X1941", "X1942",  "X1943", "X1944", "X1945", "X1946", "X1947", "X1948", "X1949",  "X1950", "X1951", "X1952", "X1953", "X1954"), terms = General.Motors_invest ~      General.Motors_value + General.Motors_capital)'''


sur.equ4 = Bunch()
sur.equ4.eqnLabel = 'US.Steel'
sur.equ4.method = 'SUR'
sur.equ4.residuals = np.array([
     -37.61318, 55.01723, 39.1423, -114.7541, -184.8984, -61.92184, 
     25.05948, 9.816803, -53.0501, -97.28717, -107.0911, 58.07571, 
     42.37924, 121.5873, 10.87083, 20.8352, 121.7198, 157.128, 101.9611, 
     -106.977
    ]).reshape(20,1, order='F')
sur.equ4.params = np.array([
     62.25631, 0.1214024, 0.3691114
    ]).reshape(3,1, order='F')
sur.equ4.cov_params = np.array([
     11369.52, -5.187909, -2.525447, -5.187909, 0.002739435, -0.0007250238, 
     -2.525447, -0.0007250238, 0.0134136
    ]).reshape(3,3, order='F')
sur.equ4.cov_params_rownames = ['(Intercept)', 'value', 'capital', ]
sur.equ4.cov_params_colnames = ['(Intercept)', 'value', 'capital', ]
sur.equ4.fittedvalues = np.array([
     247.5132, 300.2828, 430.7577, 377.0541, 415.2984, 423.5218, 447.7405, 
     435.7832, 414.6501, 385.4872, 365.7911, 362.2243, 378.1208, 372.9127, 
     394.2292, 397.9648, 466.4802, 488.372, 539.0389, 566.277
    ]).reshape(20,1, order='F')
sur.equ4.terms = '''US.Steel_invest ~ US.Steel_value + US.Steel_capital'''
sur.equ4.rank = 3
sur.equ4.nCoef_sys = 15
sur.equ4.rank_sys = 15
sur.equ4.df_resid = 17
sur.equ4.df_resid_sys = 85
sur.equ4.model = '''structure(list(US.Steel_invest = c(209.9, 355.3, 469.9, 262.3,  230.4, 361.6, 472.8, 445.6, 361.6, 288.2, 258.7, 420.3, 420.5,  494.5, 405.1, 418.8, 588.2, 645.5, 641, 459.3), US.Steel_value = c(1362.4,  1807.1, 2676.3, 1801.9, 1957.3, 2202.9, 2380.5, 2168.6, 1985.1,  1813.9, 1850.2, 2067.7, 1796.7, 1625.8, 1667, 1677.4, 2289.5,  2159.4, 2031.3, 2115.5), US.Steel_capital = c(53.8, 50.5, 118.1,  260.2, 312.7, 254.2, 261.4, 298.7, 301.8, 279.1, 213.8, 132.6,  264.8, 306.9, 351.1, 357.8, 342.1, 444.2, 623.6, 669.7)), .Names = c("US.Steel_invest",  "US.Steel_value", "US.Steel_capital"), class = "data.frame", row.names = c("X1935",  "X1936", "X1937", "X1938", "X1939", "X1940", "X1941", "X1942",  "X1943", "X1944", "X1945", "X1946", "X1947", "X1948", "X1949",  "X1950", "X1951", "X1952", "X1953", "X1954"), terms = US.Steel_invest ~      US.Steel_value + US.Steel_capital)'''


sur.equ5 = Bunch()
sur.equ5.eqnLabel = 'Westinghouse'
sur.equ5.method = 'SUR'
sur.equ5.residuals = np.array([
     0.6530944, -4.621561, -7.758567, -10.87598, -12.87523, -9.394208, 
     15.28059, 7.697017, -2.791415, -2.828532, -7.647365, 5.504111, 
     16.62065, 5.224848, -8.351701, -10.8296, 6.617559, 15.43294, 13.91221, 
     -8.968863
    ]).reshape(20,1, order='F')
sur.equ5.params = np.array([
     1.407487, 0.05635611, 0.04290209
    ]).reshape(3,1, order='F')
sur.equ5.cov_params = np.array([
     39.2104, -0.06063476, 0.0689298, -0.06063476, 0.0001316823, 
     -0.0003235898, 0.0689298, -0.0003235898, 0.001730147
    ]).reshape(3,3, order='F')
sur.equ5.cov_params_rownames = ['(Intercept)', 'value', 'capital', ]
sur.equ5.cov_params_colnames = ['(Intercept)', 'value', 'capital', ]
sur.equ5.fittedvalues = np.array([
     12.27691, 30.52156, 42.80857, 33.76598, 31.71523, 37.96421, 33.22941, 
     35.64298, 39.81141, 40.63853, 46.91736, 47.95589, 38.93935, 44.33515, 
     40.3917, 43.0696, 47.76244, 56.34706, 76.16779, 77.56886
    ]).reshape(20,1, order='F')
sur.equ5.terms = '''Westinghouse_invest ~ Westinghouse_value + Westinghouse_capital'''
sur.equ5.rank = 3
sur.equ5.nCoef_sys = 15
sur.equ5.rank_sys = 15
sur.equ5.df_resid = 17
sur.equ5.df_resid_sys = 85
sur.equ5.model = '''structure(list(Westinghouse_invest = c(12.93, 25.9, 35.05, 22.89,  18.84, 28.57, 48.51, 43.34, 37.02, 37.81, 39.27, 53.46, 55.56,  49.56, 32.04, 32.24, 54.38, 71.78, 90.08, 68.6), Westinghouse_value = c(191.5,  516, 729, 560.4, 519.9, 628.5, 537.1, 561.2, 617.2, 626.7, 737.2,  760.5, 581.4, 662.3, 583.8, 635.2, 723.8, 864.1, 1193.5, 1188.9 ), Westinghouse_capital = c(1.8, 0.8, 7.4, 18.1, 23.5, 26.5,  36.2, 60.8, 84.4, 91.2, 92.4, 86, 111.1, 130.6, 141.8, 136.7,  129.7, 145.5, 174.8, 213.5)), .Names = c("Westinghouse_invest",  "Westinghouse_value", "Westinghouse_capital"), class = "data.frame", row.names = c("X1935",  "X1936", "X1937", "X1938", "X1939", "X1940", "X1941", "X1942",  "X1943", "X1944", "X1945", "X1946", "X1947", "X1948", "X1949",  "X1950", "X1951", "X1952", "X1953", "X1954"), terms = Westinghouse_invest ~      Westinghouse_value + Westinghouse_capital)'''
sur.fittedvalues = np.concatenate((sur.equ1.fittedvalues,sur.equ2.fittedvalues,
                                sur.equ3.fittedvalues,sur.equ4.fittedvalues,
                                sur.equ5.fittedvalues))

