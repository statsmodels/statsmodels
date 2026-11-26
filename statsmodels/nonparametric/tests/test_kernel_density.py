import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
import pytest

import statsmodels.api as sm
from statsmodels.nonparametric._kernel_base import gpke
from statsmodels.nonparametric.kernel_density import KDEMultivariate

nparam = sm.nonparametric


class KDETestBase:
    def setup_method(self):
        nobs = 60
        rs = np.random.RandomState(123456)
        self.o = rs.binomial(2, 0.7, size=(nobs, 1))
        self.o2 = rs.binomial(3, 0.7, size=(nobs, 1))
        self.c1 = rs.normal(size=(nobs, 1))
        self.c2 = rs.normal(10, 1, size=(nobs, 1))
        self.c3 = rs.normal(10, 2, size=(nobs, 1))
        self.noise = rs.normal(size=(nobs, 1))
        b0 = 0.3
        b1 = 1.2
        b2 = 3.7  # regression coefficients
        self.y = b0 + b1 * self.c1 + b2 * self.c2 + self.noise
        self.y2 = b0 + b1 * self.c1 + b2 * self.c2 + self.o + self.noise
        # Italy data from R's np package (the first 50 obs) R>> data (Italy)

        self.Italy_gdp = [
            8.556,
            12.262,
            9.587,
            8.119,
            5.537,
            6.796,
            8.638,
            6.483,
            6.212,
            5.111,
            6.001,
            7.027,
            4.616,
            3.922,
            4.688,
            3.957,
            3.159,
            3.763,
            3.829,
            5.242,
            6.275,
            8.518,
            11.542,
            9.348,
            8.02,
            5.527,
            6.865,
            8.666,
            6.672,
            6.289,
            5.286,
            6.271,
            7.94,
            4.72,
            4.357,
            4.672,
            3.883,
            3.065,
            3.489,
            3.635,
            5.443,
            6.302,
            9.054,
            12.485,
            9.896,
            8.33,
            6.161,
            7.055,
            8.717,
            6.95,
        ]

        self.Italy_year = [
            1951,
            1951,
            1951,
            1951,
            1951,
            1951,
            1951,
            1951,
            1951,
            1951,
            1951,
            1951,
            1951,
            1951,
            1951,
            1951,
            1951,
            1951,
            1951,
            1951,
            1951,
            1952,
            1952,
            1952,
            1952,
            1952,
            1952,
            1952,
            1952,
            1952,
            1952,
            1952,
            1952,
            1952,
            1952,
            1952,
            1952,
            1952,
            1952,
            1952,
            1952,
            1952,
            1953,
            1953,
            1953,
            1953,
            1953,
            1953,
            1953,
            1953,
        ]

        # OECD panel data from NP  R>> data(oecdpanel)
        self.growth = [
            -0.0017584,
            0.00740688,
            0.03424461,
            0.03848719,
            0.02932506,
            0.03769199,
            0.0466038,
            0.00199456,
            0.03679607,
            0.01917304,
            -0.00221,
            0.00787269,
            0.03441118,
            -0.0109228,
            0.02043064,
            -0.0307962,
            0.02008947,
            0.00580313,
            0.00344502,
            0.04706358,
            0.03585851,
            0.01464953,
            0.04525762,
            0.04109222,
            -0.0087903,
            0.04087915,
            0.04551403,
            0.036916,
            0.00369293,
            0.0718669,
            0.02577732,
            -0.0130759,
            -0.01656641,
            0.00676429,
            0.08833017,
            0.05092105,
            0.02005877,
            0.00183858,
            0.03903173,
            0.05832116,
            0.0494571,
            0.02078484,
            0.09213897,
            0.0070534,
            0.08677202,
            0.06830603,
            -0.00041,
            0.0002856,
            0.03421225,
            -0.0036825,
        ]

        self.oecd = [
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]

        self.weights = rs.random(nobs)
        self.weights_orig = self.weights.copy()


class TestKDEUnivariate(KDETestBase):

    def test_pdf_non_fft(self):

        kde = nparam.KDEUnivariate(self.noise)
        kde.fit(fft=False, bw="scott")

        grid = kde.support
        testx = [grid[10 * i] for i in range(6)]

        # Test against values from R 'ks' package
        kde_expected = [
            0.00016808277984236013,
            0.030759614592368954,
            0.14123404934759243,
            0.28807147408162409,
            0.25594519303876273,
            0.056593973915651047,
        ]

        kde_vals0 = kde.density[10 * np.arange(6)]
        kde_vals = kde.evaluate(testx)

        npt.assert_allclose(kde_vals, kde_expected, atol=1e-6)
        npt.assert_allclose(kde_vals0, kde_expected, atol=1e-6)

    def test_weighted_pdf_non_fft(self):

        kde = nparam.KDEUnivariate(self.noise)
        weights = self.weights_orig.copy()
        kde.fit(weights=weights, fft=False, bw="scott")
        npt.assert_allclose(weights, self.weights_orig)

        grid = kde.support
        testx = np.array([grid[10 * i] for i in range(6)])

        # Test against values from R 'ks' package
        kde_expected = [
            9.1998858033950757e-05,
            0.018761981151370496,
            0.14425925509365087,
            0.30307631742267443,
            0.2405445849994125,
            0.06433170684797665,
        ]

        kde_vals0 = kde.density[10 * np.arange(6)]
        kde_vals = kde.evaluate(testx)

        npt.assert_allclose(kde_vals, kde_expected, atol=1e-6)
        npt.assert_allclose(kde_vals0, kde_expected, atol=1e-6)

    def test_all_samples_same_location_bw(self):
        x = np.ones(100)
        kde = nparam.KDEUnivariate(x)
        with pytest.raises(RuntimeError, match="Selected KDE bandwidth is 0"):
            kde.fit()

    def test_int(self):
        rs = np.random.RandomState(3321839021)
        x = rs.randint(0, 100, size=1000)
        kde = nparam.KDEUnivariate(x)
        kde.fit()

        kde_double = nparam.KDEUnivariate(x.astype("double"))
        kde_double.fit()

        assert_allclose(kde.bw, kde_double.bw)


class TestKDEMultivariate(KDETestBase):
    @pytest.mark.slow
    def test_pdf_mixeddata_CV_LS(self):
        with pytest.warns(FutureWarning, match="After 0.17"):
            dens_u = nparam.KDEMultivariate(
                data=[self.c1, self.o, self.o2], var_type="coo", bw="cv_ls"
            )
        npt.assert_allclose(dens_u.bw, [0.70949395, 0.08765272, 0.09211527], atol=1e-6)

        # Matches R to 3 decimals; results seem more stable than with R.
        # Can be checked with following code:
        # import rpy2.robjects as robjects
        # from rpy2.robjects.packages import importr
        # NP = importr('np')
        # r = robjects.r
        # D = {"S1": robjects.FloatVector(c1), "S2":robjects.FloatVector(c2),
        #      "S3":robjects.FloatVector(c3), "S4":robjects.FactorVector(o),
        #      "S5":robjects.FactorVector(o2)}
        # df = robjects.DataFrame(D)
        # formula = r('~S1+ordered(S4)+ordered(S5)')
        # r_bw = NP.npudensbw(formula, data=df, bwmethod='cv.ls')

    @pytest.mark.slow
    def test_pdf_mixeddata_LS_vs_ML(self):
        with pytest.warns(FutureWarning, match="After 0.17"):
            dens_ls = nparam.KDEMultivariate(
                data=[self.c1, self.o, self.o2], var_type="coo", bw="cv_ls"
            )
        with pytest.warns(FutureWarning, match="After 0.17"):
            dens_ml = nparam.KDEMultivariate(
                data=[self.c1, self.o, self.o2], var_type="coo", bw="cv_ml"
            )
        npt.assert_allclose(dens_ls.bw, dens_ml.bw, atol=0, rtol=0.5)

    def test_pdf_mixeddata_CV_ML(self):
        # Test ML cross-validation
        with pytest.warns(FutureWarning, match="After 0.17"):
            dens_ml = nparam.KDEMultivariate(
                data=[self.c1, self.o, self.c2], var_type="coc", bw="cv_ml"
            )
        R_bw = [1.021563, 2.806409e-14, 0.5142077]
        npt.assert_allclose(dens_ml.bw, R_bw, atol=0.1, rtol=0.1)

    @pytest.mark.slow
    def test_pdf_continuous(self):
        # Test for only continuous data
        with pytest.warns(FutureWarning, match="After 0.17"):
            dens = nparam.KDEMultivariate(
                data=[self.growth, self.Italy_gdp], var_type="cc", bw="cv_ls"
            )
        # take the first data points from the training set
        sm_result = np.squeeze(dens.pdf()[0:5])
        R_result = [1.6202284, 0.7914245, 1.6084174, 2.4987204, 1.3705258]

        # CODE TO REPRODUCE THE RESULTS IN R
        # library(np)
        # data(oecdpanel)
        # data (Italy)
        # bw <-npudensbw(formula = ~oecdpanel$growth[1:50] + Italy$gdp[1:50],
        # bwmethod ='cv.ls')
        # fhat <- fitted(npudens(bws=bw))
        # fhat[1:5]
        npt.assert_allclose(sm_result, R_result, atol=1e-3)

    def test_pdf_ordered(self):
        # Test for only ordered data
        with pytest.warns(FutureWarning, match="After 0.17"):
            dens = nparam.KDEMultivariate(data=[self.oecd], var_type="o", bw="cv_ls")
        sm_result = np.squeeze(dens.pdf()[0:5])
        R_result = [0.7236395, 0.7236395, 0.2763605, 0.2763605, 0.7236395]
        # lower tol here. only 2nd decimal
        npt.assert_allclose(sm_result, R_result, atol=1e-1)

    @pytest.mark.slow
    def test_unordered_CV_LS(self):
        with pytest.warns(FutureWarning, match="After 0.17"):
            dens = nparam.KDEMultivariate(
                data=[self.growth, self.oecd], var_type="cu", bw="cv_ls"
            )
        R_result = [0.0052051, 0.05835941]
        npt.assert_allclose(dens.bw, R_result, atol=1e-2)

    def test_continuous_cdf(self):
        with pytest.warns(FutureWarning, match="After 0.17"):
            dens = nparam.KDEMultivariate(
                data=[self.Italy_gdp, self.growth], var_type="cc", bw="cv_ml"
            )
        sm_result = dens.cdf()[0:5]
        R_result = [0.192180770, 0.299505196, 0.557303666, 0.513387712, 0.210985350]
        npt.assert_allclose(sm_result, R_result, atol=1e-3)

    def test_mixeddata_cdf(self):
        with pytest.warns(FutureWarning, match="After 0.17"):
            dens = nparam.KDEMultivariate(
                data=[self.Italy_gdp, self.oecd], var_type="cu", bw="cv_ml"
            )
        sm_result = dens.cdf()[0:5]
        R_result = [0.54700010, 0.65907039, 0.89676865, 0.74132941, 0.25291361]
        npt.assert_allclose(sm_result, R_result, atol=1e-3)

    @pytest.mark.joblib
    @pytest.mark.slow
    def test_continuous_cvls_efficient(self):
        nobs = 400
        rs = np.random.RandomState(12345)
        C1 = rs.normal(size=(nobs,))
        C2 = rs.normal(2, 1, size=(nobs,))
        Y = 0.3 + 1.2 * C1 - 0.9 * C2
        with pytest.warns(FutureWarning, match="After 0.17"):
            dens_efficient = nparam.KDEMultivariate(
                data=[Y, C1],
                var_type="cc",
                bw="cv_ls",
                defaults=nparam.EstimatorSettings(efficient=True, n_sub=100, n_jobs=2),
            )
        # dens = nparam.KDEMultivariate(data=[Y, C1], var_type='cc', bw='cv_ls',
        #                  defaults=nparam.EstimatorSettings(efficient=False))
        # bw = dens.bw
        bw = np.array([0.3404, 0.1666])
        npt.assert_allclose(bw, dens_efficient.bw, atol=0.1, rtol=0.2)

    @pytest.mark.joblib
    @pytest.mark.slow
    def test_continuous_cvml_efficient(self):
        nobs = 400
        rs = np.random.RandomState(12345)
        C1 = rs.normal(size=(nobs,))
        C2 = rs.normal(2, 1, size=(nobs,))
        Y = 0.3 + 1.2 * C1 - 0.9 * C2

        with pytest.warns(FutureWarning, match="After 0.17"):
            dens_efficient = nparam.KDEMultivariate(
                data=[Y, C1],
                var_type="cc",
                bw="cv_ml",
                defaults=nparam.EstimatorSettings(efficient=True, n_sub=100, n_jobs=2),
            )
        # dens = nparam.KDEMultivariate(data=[Y, C1], var_type='cc', bw='cv_ml',
        #                  defaults=nparam.EstimatorSettings(efficient=False))
        # bw = dens.bw
        bw = np.array([0.4471, 0.2861])
        npt.assert_allclose(bw, dens_efficient.bw, atol=0.1, rtol=0.2)

    @pytest.mark.joblib
    @pytest.mark.slow
    def test_efficient_notrandom(self):
        nobs = 400
        rs = np.random.RandomState(12345)
        C1 = rs.normal(size=(nobs,))
        C2 = rs.normal(2, 1, size=(nobs,))
        Y = 0.3 + 1.2 * C1 - 0.9 * C2

        with pytest.warns(FutureWarning, match="After 0.17"):
            dens_efficient = nparam.KDEMultivariate(
                data=[Y, C1],
                var_type="cc",
                bw="cv_ml",
                defaults=nparam.EstimatorSettings(
                    efficient=True, randomize=False, n_sub=100, n_jobs=2
                ),
            )
        with pytest.warns(FutureWarning, match="After 0.17"):
            dens = nparam.KDEMultivariate(data=[Y, C1], var_type="cc", bw="cv_ml")
        npt.assert_allclose(dens.bw, dens_efficient.bw, atol=0.1, rtol=0.2)

    def test_efficient_user_specified_bw(self):
        nobs = 400
        rs = np.random.RandomState(12345)
        C1 = rs.normal(size=(nobs,))
        C2 = rs.normal(2, 1, size=(nobs,))
        bw_user = [0.23, 434697.22]

        with pytest.warns(FutureWarning, match="After 0.17"):
            dens = nparam.KDEMultivariate(
                data=[C1, C2],
                var_type="cc",
                bw=bw_user,
                defaults=nparam.EstimatorSettings(
                    efficient=True, randomize=False, n_sub=100
                ),
            )
        npt.assert_equal(dens.bw, bw_user)

    @pytest.mark.parametrize("efficient", [False, True])
    def test_scalar_user_specified_bw(self, efficient):
        # GH4747 a scalar user-specified bandwidth is promoted to 1-D, so
        # that it behaves the same as the equivalent length-one sequence.
        rs = np.random.RandomState(12345)
        c1 = rs.normal(size=(60,))

        dens_scalar = nparam.KDEMultivariate(
            data=[c1],
            var_type="c",
            bw=0.5,
            defaults=nparam.EstimatorSettings(efficient=efficient),
            rng=12345,
        )
        dens_seq = nparam.KDEMultivariate(
            data=[c1],
            var_type="c",
            bw=[0.5],
            defaults=nparam.EstimatorSettings(efficient=efficient),
            rng=12345,
        )

        npt.assert_equal(dens_scalar.bw, np.array([0.5]))
        npt.assert_allclose(dens_scalar.pdf(), dens_seq.pdf())

    @pytest.mark.parametrize("efficient", [False, True])
    def test_2d_user_specified_bw_raises(self, efficient):
        # GH4747 the bandwidth is limited to a 1-D array
        rs = np.random.RandomState(12345)
        c1 = rs.normal(size=(60,))
        with pytest.raises(ValueError, match="ndim"):
            nparam.KDEMultivariate(
                data=[c1],
                var_type="c",
                bw=np.full((2, 2), 0.5),
                defaults=nparam.EstimatorSettings(efficient=efficient),
                rng=12345,
            )


class TestKDEMultivariateConditional(KDETestBase):
    @pytest.mark.slow
    def test_mixeddata_CV_LS(self):
        with pytest.warns(FutureWarning, match="After 0.17"):
            dens_ls = nparam.KDEMultivariateConditional(
                endog=[self.Italy_gdp],
                exog=[self.Italy_year],
                dep_type="c",
                indep_type="o",
                bw="cv_ls",
            )
        # R result: [1.6448, 0.2317373]
        R_bw = [1.6448, 0.2317373]
        npt.assert_allclose(dens_ls.bw, R_bw, atol=1e-3)

    def test_continuous_CV_ML(self):
        with pytest.warns(FutureWarning, match="After 0.17"):
            dens_ml = nparam.KDEMultivariateConditional(
                endog=[self.Italy_gdp],
                exog=[self.growth],
                dep_type="c",
                indep_type="c",
                bw="cv_ml",
            )
        # Results from R
        npt.assert_allclose(dens_ml.bw, [0.5341164, 0.04510836], atol=1e-3)

    @pytest.mark.slow
    def test_unordered_CV_LS(self):
        with pytest.warns(FutureWarning, match="After 0.17"):
            nparam.KDEMultivariateConditional(
                endog=[self.oecd],
                exog=[self.growth],
                dep_type="u",
                indep_type="c",
                bw="cv_ls",
            )
        # TODO: assert missing

    def test_pdf_continuous(self):
        # Hardcode here the bw that will be calculated is we had used
        # ``bw='cv_ml'``.  That calculation is slow, and tested in other tests.
        bw_cv_ml = np.array([0.010043, 12095254.7])  # TODO: odd numbers (?!)
        with pytest.warns(FutureWarning, match="After 0.17"):
            dens = nparam.KDEMultivariateConditional(
                endog=[self.growth],
                exog=[self.Italy_gdp],
                dep_type="c",
                indep_type="c",
                bw=bw_cv_ml,
            )
        sm_result = np.squeeze(dens.pdf()[0:5])
        R_result = [11.97964, 12.73290, 13.23037, 13.46438, 12.22779]
        npt.assert_allclose(sm_result, R_result, atol=1e-3)

    @pytest.mark.slow
    def test_pdf_mixeddata(self):
        with pytest.warns(FutureWarning, match="After 0.17"):
            dens = nparam.KDEMultivariateConditional(
                endog=[self.Italy_gdp],
                exog=[self.Italy_year],
                dep_type="c",
                indep_type="o",
                bw="cv_ls",
            )
        sm_result = np.squeeze(dens.pdf()[0:5])
        expected = [0.08469226, 0.01737731, 0.05679909, 0.09744726, 0.15086674]

        # CODE TO REPRODUCE IN R
        # library(np)
        # data (Italy)
        # bw <- npcdensbw(formula =
        # Italy$gdp[1:50]~ordered(Italy$year[1:50]),bwmethod='cv.ls')
        # fhat <- fitted(npcdens(bws=bw))
        # fhat[1:5]
        npt.assert_allclose(sm_result, expected, atol=1e-3)

    def test_continuous_normal_ref(self):
        # test for normal reference rule of thumb with continuous data
        with pytest.warns(FutureWarning, match="After 0.17"):
            dens_nm = nparam.KDEMultivariateConditional(
                endog=[self.Italy_gdp],
                exog=[self.growth],
                dep_type="c",
                indep_type="c",
                bw="normal_reference",
            )
        sm_result = dens_nm.bw
        R_result = [1.283532, 0.01535401]
        # TODO: here we need a smaller tolerance.check!
        npt.assert_allclose(sm_result, R_result, atol=1e-1)

        # test default bandwidth method, should be normal_reference
        with pytest.warns(FutureWarning, match="After 0.17"):
            dens_nm2 = nparam.KDEMultivariateConditional(
                endog=[self.Italy_gdp],
                exog=[self.growth],
                dep_type="c",
                indep_type="c",
                bw=None,
            )

        assert_allclose(dens_nm2.bw, dens_nm.bw, rtol=1e-10)
        assert_equal(dens_nm2._bw_method, "normal_reference")
        # check repr works #3125
        repr(dens_nm2)

    def test_continuous_cdf(self):
        with pytest.warns(FutureWarning, match="After 0.17"):

            dens_nm = nparam.KDEMultivariateConditional(
                endog=[self.Italy_gdp],
                exog=[self.growth],
                dep_type="c",
                indep_type="c",
                bw="normal_reference",
            )
        sm_result = dens_nm.cdf()[0:5]
        R_result = [0.81304920, 0.95046942, 0.86878727, 0.71961748, 0.38685423]
        npt.assert_allclose(sm_result, R_result, atol=1e-3)

    @pytest.mark.slow
    def test_mixeddata_cdf(self):
        with pytest.warns(FutureWarning, match="After 0.17"):
            dens = nparam.KDEMultivariateConditional(
                endog=[self.Italy_gdp],
                exog=[self.Italy_year],
                dep_type="c",
                indep_type="o",
                bw="cv_ls",
            )
        sm_result = dens.cdf()[0:5]
        expected = [0.8118257, 0.9724863, 0.8843773, 0.7720359, 0.4361867]
        npt.assert_allclose(sm_result, expected, atol=1e-3)

    @pytest.mark.joblib
    @pytest.mark.slow
    def test_continuous_cvml_efficient(self):
        nobs = 500
        rs = np.random.RandomState(12345)
        ovals = rs.binomial(2, 0.5, size=(nobs,))
        C1 = rs.normal(size=(nobs,))
        noise = rs.normal(size=(nobs,))
        b0 = 3
        b1 = 1.2
        b2 = 3.7  # regression coefficients
        Y = b0 + b1 * C1 + b2 * ovals + noise

        dens_efficient = nparam.KDEMultivariateConditional(
            endog=[Y],
            exog=[C1],
            dep_type="c",
            indep_type="c",
            bw="cv_ml",
            defaults=nparam.EstimatorSettings(efficient=True, n_sub=50, n_jobs=2),
            rng=12345,
        )

        # dens = nparam.KDEMultivariateConditional(endog=[Y], exog=[C1],
        #                   dep_type='c', indep_type='c', bw='cv_ml')
        # bw = dens.bw
        bw_expected = np.array([0.73387, 0.43715])
        npt.assert_allclose(dens_efficient.bw, bw_expected, atol=0, rtol=1e-3)

    def test_efficient_user_specified_bw(self):
        nobs = 400
        rs = np.random.RandomState(12345)
        C1 = rs.normal(size=(nobs,))
        C2 = rs.normal(2, 1, size=(nobs,))
        bw_user = [0.23, 434697.22]
        with pytest.warns(FutureWarning, match="After 0.17"):
            dens = nparam.KDEMultivariate(
                data=[C1, C2],
                var_type="cc",
                bw=bw_user,
                defaults=nparam.EstimatorSettings(
                    efficient=True, randomize=False, n_sub=100
                ),
            )
        npt.assert_equal(dens.bw, bw_user)


@pytest.mark.parametrize(
    "dep_type, indep_type", [("c", "co"), ("c", "cu"), ("o", "c"), ("cu", "cou")]
)
def test_conditional_cdf_formula(dep_type, indep_type):
    # GH 6076: cdf is sum(G * W) / sum(W) with the product kernel cdf G of
    # endog and the product kernel W of exog
    rng = np.random.default_rng(6076)
    nobs = 50

    def draw(var_type):
        # a separate column for every variable
        draws = {
            "c": lambda: rng.normal(size=nobs),
            "o": lambda: rng.integers(0, 4, nobs),
            "u": lambda: rng.integers(0, 3, nobs),
        }
        return np.column_stack([draws[t]() for t in var_type])

    endog = draw(dep_type)
    exog = draw(indep_type)
    bw = np.full(len(dep_type) + len(indep_type), 0.4)
    dens = nparam.KDEMultivariateConditional(
        endog=endog,
        exog=exog,
        dep_type=dep_type,
        indep_type=indep_type,
        bw=bw,
        rng=0,
    )
    k_dep = len(dep_type)
    expected = np.empty(nobs)
    for i in range(nobs):
        G = gpke(
            bw[:k_dep],
            data=endog,
            data_predict=endog[i],
            var_type=dep_type,
            ckertype="gaussian_cdf",
            ukertype="aitchisonaitken_cdf",
            okertype="wangryzin_cdf",
            tosum=False,
        )
        W = gpke(
            bw[k_dep:],
            data=exog,
            data_predict=exog[i],
            var_type=indep_type,
            tosum=False,
        )
        mu_x = gpke(bw[k_dep:], data=exog, data_predict=exog[i], var_type=indep_type)
        expected[i] = (G * W).sum() / mu_x
    assert_allclose(dens.cdf(), expected, rtol=1e-12)


def test_normal_reference_bw_of_unordered_variable_in_range():
    # GH 3790: the normal reference rule of thumb is defined for continuous
    # variables, so applying it to an unordered variable makes the bandwidth
    # scale with the arbitrary numeric codes.  The bandwidth of an unordered
    # variable is the smoothing probability of the Aitchison-Aitken kernel
    # (see kernels.aitchison_aitken), for which the estimate is a probability
    # density only if 0 <= bw <= (levels - 1) / levels.  Larger values put
    # less weight on the observed levels than on the unobserved ones, which
    # made the density negative for codes on a large scale.
    codes = 1000 * np.array([0, 0, 0, 0, 1])
    dens = nparam.KDEMultivariate(data=codes, var_type="u", rng=0)
    # two levels, so the largest valid bandwidth is 1 / 2
    npt.assert_allclose(dens.bw, [0.5])
    pdf = dens.pdf([0, 1000])
    assert np.all(np.isfinite(pdf))
    assert np.all(pdf >= 0)
    # the bandwidth is at its bound, so the kernel is the uniform one
    npt.assert_allclose(pdf, [0.5, 0.5])
    npt.assert_allclose(pdf.sum(), 1.0)

    # codes that give a bandwidth inside the valid range are unchanged
    dens_small = nparam.KDEMultivariate(data=[0, 0, 0, 0, 1], var_type="u", rng=0)
    npt.assert_allclose(dens_small.bw, [0.30730658])
    npt.assert_allclose(dens_small.pdf([0, 1]), [0.61561605, 0.38438395], rtol=1e-6)


def test_normal_reference_bw_of_unordered_variable_mixed_data():
    # The bound is applied to the unordered variables only, the continuous
    # bandwidth is still the normal reference rule of thumb.
    codes = 1000 * np.array([0, 0, 0, 0, 1])
    cont = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    dens = nparam.KDEMultivariate(
        data=np.column_stack((codes, cont)), var_type="uc", rng=0
    )
    npt.assert_allclose(dens.bw[0], 0.5)
    npt.assert_allclose(dens.bw[1], 1.06 * np.std(cont) * 5 ** (-1.0 / (4 + 2)))
    pdf = dens.pdf([[0, 0.1], [1000, 0.5]])
    assert np.all(np.isfinite(pdf))
    assert np.all(pdf > 0)


def test_normal_reference_bw_of_unordered_dependent_variable():
    # KDEMultivariateConditional selects its bandwidths with the same rule
    codes = 1000 * np.array([0, 0, 0, 0, 1])
    cont = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    dens = nparam.KDEMultivariateConditional(
        endog=codes,
        exog=cont,
        dep_type="u",
        indep_type="c",
        bw="normal_reference",
        rng=0,
    )
    npt.assert_allclose(dens.bw[0], 0.5)
    pdf = dens.pdf(endog_predict=[0, 1000], exog_predict=[0.1, 0.5])
    assert np.all(np.isfinite(pdf))
    assert np.all(pdf > 0)


@pytest.mark.parametrize("scale", [1, 100])
def test_normal_reference_bw_of_ordered_variable_below_one(scale):
    # The Wang-Ryzin kernel contains the factor 1 - bw, so that the density
    # is identically zero at bw = 1.  The rule of thumb 1.06 * std * nobs ** -0.2
    # is 1.3 for these codes and larger for the scaled codes, so the bandwidth
    # has to be bounded strictly below 1.
    codes = np.repeat(np.arange(11), 10) * scale
    dens = KDEMultivariate(data=codes, var_type="o", rng=0)
    assert 0 < dens.bw[0] < 1
    pdf = dens.pdf(np.unique(codes))
    assert np.all(np.isfinite(pdf))
    assert np.all(pdf > 0)


@pytest.mark.parametrize("bw", ["cv_ml", "cv_ls"])
@pytest.mark.parametrize(
    "levels, r_bw", [(3, 0.66000057), (4, 0.74250011), (5, 0.79200058), (6, 0.8250008)]
)
def test_cv_bw_of_unordered_variable_at_bound(levels, r_bw, bw):
    # The leave-one-out likelihood of balanced data increases with the
    # bandwidth up to the fully smoothed, uniform kernel, which is the bound
    # (levels - 1) / levels.  The cross-validation returned the invalid
    # bandwidth 1 because the search was not bounded.
    # Reference value of the cv_ml bandwidth, np 0.70.5 with R 4.6.1,
    # np backs off by 1% from its bound:
    #   library(np)
    #   x <- factor(rep(0:(levels - 1), each = 20))
    #   npudensbw(dat = data.frame(x = x), bwmethod = "cv.ml",
    #             ukertype = "aitchisonaitken")$bw
    x = np.repeat(np.arange(levels), 20)
    bound = (levels - 1) / levels
    dens = KDEMultivariate(data=x, var_type="u", bw=bw, rng=0)
    assert 0.99 * bound < dens.bw[0] <= bound
    if bw == "cv_ml":
        assert_allclose(dens.bw, r_bw, rtol=0.02)


def test_cv_ml_bw_of_ordered_variable_matches_r():
    # The search started at the rule of thumb 1.31, outside of the kernel's
    # range, and ended in the bandwidth 1, where the density is zero and the
    # leave-one-out likelihood is infinite.
    # Reference value, np 0.70.5 with R 4.6.1:
    #   library(np)
    #   x <- ordered(rep(0:10, each = 10), levels = 0:10)
    #   npudensbw(dat = data.frame(x = x), bwmethod = "cv.ml",
    #             okertype = "wangvanryzin")$bw
    x = np.repeat(np.arange(11), 10)
    dens = KDEMultivariate(data=x, var_type="o", bw="cv_ml", rng=0)
    assert_allclose(dens.bw, 0.038873737, atol=1e-3)
    assert np.isfinite(dens.loo_likelihood(dens.bw, np.log))


def test_cv_ml_bw_is_constrained_optimum():
    # With the unordered bandwidth at its bound (2 / 3), the best continuous
    # bandwidth is not the one of the unconstrained search clipped to the
    # bound: the leave-one-out log likelihood is 0.86 lower at that bandwidth.
    rng = np.random.default_rng(26)
    data = np.column_stack((rng.integers(0, 3, 40), rng.standard_normal(40)))
    dens = KDEMultivariate(data=data, var_type="uc", bw="cv_ml", rng=0)
    assert dens.bw[0] <= 2 / 3
    grid = [
        dens.loo_likelihood(np.array([u, c]), np.log)
        for u in np.linspace(0.05, 2 / 3, 14)
        for c in np.linspace(0.05, 1.0, 20)
    ]
    assert dens.loo_likelihood(dens.bw, np.log) < min(grid) + 1e-3


@pytest.mark.parametrize("var_type, upper", [("u", 0.75), ("o", 1.0)])
def test_efficient_normal_reference_bw_of_discrete_variable_in_range(var_type, upper):
    # The codes are on a large scale, so the bandwidths of the sub samples,
    # which the efficient estimation scales up to the full sample, are far
    # outside of the range of the kernel if they are not bounded.
    rng = np.random.default_rng(3790)
    data = np.column_stack((rng.standard_normal(300), 1000 * rng.integers(0, 4, 300)))
    dens = KDEMultivariate(
        data=data,
        var_type="c" + var_type,
        defaults=nparam.EstimatorSettings(efficient=True, n_sub=50, n_jobs=1),
        rng=0,
    )
    assert 0 < dens.bw[1] <= upper
    if var_type == "o":
        assert dens.bw[1] < 1
    assert np.all(np.isfinite(dens.pdf(data[:5])))


@pytest.mark.parametrize("bw", ["cv_ml", "cv_ls"])
def test_efficient_cv_bw_of_unordered_variable_scales_bound(bw):
    # The balanced sub samples of 60 observations have the bandwidth 3 / 4,
    # the bound of 4 levels, and the full sample of 300 observations has the
    # bandwidth 3 / 4 * (60 / 300) ** (2 / 4)
    dens = KDEMultivariate(
        data=np.tile(np.arange(4), 75),
        var_type="u",
        bw=bw,
        defaults=nparam.EstimatorSettings(efficient=True, n_sub=60, n_jobs=1),
        rng=0,
    )
    assert_allclose(dens.bw, 0.75 * (60 / 300) ** 0.5, rtol=1e-2)


@pytest.mark.parametrize("bw", ["cv_ml", "cv_ls"])
def test_cv_bw_of_unordered_dependent_variable_in_range(bw):
    # KDEMultivariateConditional: the dependent variable is independent of
    # the continuous one, so the best bandwidth of the unordered variable is
    # the largest valid one.
    rng = np.random.default_rng(3790)
    endog = 1000 * np.repeat(np.arange(3), 20)
    exog = rng.standard_normal(60)
    dens = nparam.KDEMultivariateConditional(
        endog=endog, exog=exog, dep_type="u", indep_type="c", bw=bw, rng=0
    )
    assert 0 < dens.bw[0] <= 2 / 3


@pytest.mark.filterwarnings("ignore:divide by zero encountered in divide")
@pytest.mark.parametrize("bw", ["normal_reference", "cv_ml", "cv_ls"])
def test_bw_of_single_level_unordered_variable(bw):
    # The upper bound (levels - 1) / levels is 0 for one level, there the
    # bandwidth is bounded by 1 only
    rng = np.random.default_rng(3790)
    data = np.column_stack((np.zeros(30), rng.standard_normal(30)))
    dens = KDEMultivariate(data=data, var_type="uc", bw=bw, rng=0)
    assert np.all(np.isfinite(dens.bw))
    assert 0 <= dens.bw[0] <= 1


def test_conditional_imse_matches_hand_computed_cv():
    # KDEMultivariateConditional.imse implements the leave-one-out CV(h)
    # objective documented in its docstring; recompute it from scratch with
    # plain nested loops (rather than the vectorized kron-based
    # implementation used internally) for a tiny dataset, using the same
    # elementary kernel definitions
    # (statsmodels.nonparametric.kernels.gaussian and .gaussian_convolution).
    rng = np.random.default_rng(20250202)
    nobs = 6
    X = rng.normal(size=nobs)
    Y = 0.5 * X + rng.normal(scale=0.3, size=nobs)

    bw = np.array([0.7, 0.9])  # [h_y, h_x]
    dens = nparam.KDEMultivariateConditional(
        endog=[Y], exog=[X], dep_type="c", indep_type="c", bw=bw, rng=0
    )
    cv = dens.imse(dens.bw)

    def gaussian_k(h, a, b):
        # matches statsmodels.nonparametric.kernels.gaussian(h, Xi, x)
        return 1.0 / np.sqrt(2 * np.pi) * np.exp(-((a - b) ** 2) / (2 * h**2))

    def gaussian_conv_k(h, a, b):
        # matches kernels.gaussian_convolution(h, Xi, x)
        return 1.0 / np.sqrt(4 * np.pi) * np.exp(-((a - b) ** 2) / (4 * h**2))

    # gpke additionally divides by the product of bandwidths over the
    # continuous dimensions used in a given call (see
    # _kernel_base.gpke: ``dens = Kval.prod(axis=1)/np.prod(bw[iscontinuous])``)
    hy, hx = bw
    n = nobs
    CV = 0.0
    for ll in range(n):
        others = [i for i in range(n) if i != ll]
        Gl = 0.0
        for i in others:
            for j in others:
                Gl += (
                    (gaussian_k(hx, X[i], X[ll]) / hx)
                    * (gaussian_k(hx, X[j], X[ll]) / hx)
                    * (gaussian_conv_k(hy, Y[i], Y[j]) / hy)
                )
        Gl /= n**2

        mu_l = sum((gaussian_k(hx, X[i], X[ll]) / hx) for i in others) / n
        f_l = (
            sum(
                (gaussian_k(hy, Y[i], Y[ll]) / hy) * (gaussian_k(hx, X[i], X[ll]) / hx)
                for i in others
            )
            / n
        )
        CV += (Gl / mu_l**2) - 2 * (f_l / mu_l)
    CV /= n

    npt.assert_allclose(cv, CV, rtol=1e-8)


def _elementary_kernel(var_type, h, a, b, levels, convolution=False):
    # Kernel of a single variable at the scalar values a and b, written out
    # from the definitions in statsmodels.nonparametric.kernels. `levels` are
    # the values observed in the (leave-one-out) sample of the variable.
    if var_type == "c":
        scale = 4 if convolution else 2
        return np.exp(-((a - b) ** 2) / (scale * h**2)) / np.sqrt(scale * np.pi)

    def kernel(a, b):
        if var_type == "o":
            return 1 - h if a == b else 0.5 * (1 - h) * h ** abs(a - b)
        return 1 - h if a == b else h / (len(levels) - 1)

    if convolution:
        return sum(kernel(a, x) * kernel(b, x) for x in levels)
    return kernel(a, b)


@pytest.mark.parametrize(
    "dep_type, indep_type", [("c", "cc"), ("c", "cu"), ("o", "co"), ("cu", "cou")]
)
def test_conditional_imse_brute_force(dep_type, indep_type):
    # GH 6076: imse evaluates the kernel of the exog once for each
    # leave-one-out sample and expands it to all pairs (i, j); compare with
    # explicit loops over the pairs for several and for discrete exog
    rng = np.random.default_rng(10626)
    nobs = 7
    k_dep = len(dep_type)
    var_type = dep_type + indep_type

    def draw(var_type):
        draws = {
            "c": lambda: rng.normal(size=nobs),
            "o": lambda: rng.integers(0, 3, nobs),
            "u": lambda: rng.integers(0, 3, nobs),
        }
        return np.column_stack([draws[t]() for t in var_type])

    endog = draw(dep_type)
    exog = draw(indep_type)
    data = np.column_stack([endog, exog])
    # different bandwidths so that a mix-up of the variables changes the value
    bw = np.array(
        [0.5 + 0.1 * k if t == "c" else 0.1 + 0.05 * k for k, t in enumerate(var_type)]
    )
    dens = nparam.KDEMultivariateConditional(
        endog=endog,
        exog=exog,
        dep_type=dep_type,
        indep_type=indep_type,
        bw=bw,
        rng=0,
    )

    def product_kernel(cols, a, b, levels, convolution=False):
        # product kernel over the variables `cols`, divided by the product of
        # the bandwidths of the continuous variables
        value = 1.0
        for k in cols:
            value *= _elementary_kernel(
                var_type[k], bw[k], a[k], b[k], levels[k], convolution
            )
        return value / np.prod([bw[k] for k in cols if var_type[k] == "c"])

    dep_cols = range(k_dep)
    indep_cols = range(k_dep, len(var_type))
    all_cols = range(len(var_type))
    cv = 0.0
    for ll in range(nobs):
        others = [i for i in range(nobs) if i != ll]
        levels = [np.unique(data[others, k]) for k in all_cols]
        G = 0.0
        for i in others:
            for j in others:
                G += (
                    product_kernel(indep_cols, data[i], data[ll], levels)
                    * product_kernel(indep_cols, data[j], data[ll], levels)
                    * product_kernel(dep_cols, data[i], data[j], levels, True)
                )
        G /= nobs**2
        m_x = (
            sum(product_kernel(indep_cols, data[i], data[ll], levels) for i in others)
            / nobs
        )
        f_x_y = (
            sum(product_kernel(all_cols, data[i], data[ll], levels) for i in others)
            / nobs
        )
        cv += G / m_x**2 - 2 * f_x_y / m_x
    cv /= nobs

    assert_allclose(dens.imse(bw), cv, rtol=1e-10)


@pytest.mark.parametrize("kernel", ["biw", "cos", "epa", "gau", "tri", "triw", "uni"])
def test_all_kernels(kernel):
    rs = np.random.RandomState(32989053)
    data = rs.normal(size=200)
    x_grid = np.linspace(min(data), max(data), 200)
    density = sm.nonparametric.KDEUnivariate(data)
    density.fit(kernel="gau", fft=False)
    assert isinstance(density.evaluate(x_grid), np.ndarray)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_kdemultivariate_invalid_bw_raises():
    # negative entries previously produced negative densities, and unknown
    # string methods silently fell back to cv_ls
    data = np.tile(np.linspace(-2.0, 2.0, 50)[:, None], (1, 2))
    with pytest.raises(ValueError, match="bw must be positive"):
        KDEMultivariate(data, var_type="cc", bw=[-0.5, 0.5])
    with pytest.raises(ValueError, match="bw must be one of"):
        KDEMultivariate(data, var_type="cc", bw="bogus")
    dens = KDEMultivariate(data, var_type="cc", bw=[0.5, 0.5]).pdf([0, 0])
    assert dens > 0
