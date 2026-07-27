"""
Tests of GLSAR and diagnostics against Gretl

Created on Thu Feb 02 21:15:47 2012

Author: Josef Perktold
License: BSD-3

"""

from pathlib import Path

import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_less,
    assert_equal,
)

from statsmodels.datasets import macrodata
from statsmodels.regression.linear_model import GLSAR, OLS
import statsmodels.stats.diagnostic as smsdia
import statsmodels.stats.outliers_influence as oi
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.tools import add_constant


def compare_ftest(contrast_res, other, decimal=(5, 4)):
    assert_almost_equal(contrast_res.fvalue, other[0], decimal=decimal[0])
    assert_almost_equal(contrast_res.pvalue, other[1], decimal=decimal[1])
    assert_equal(contrast_res.df_num, other[2])
    assert_equal(contrast_res.df_denom, other[3])
    assert_equal("f", other[4])


class TestGLSARGretl:

    def test_all(self):
        d = macrodata.load_pandas().data
        gs_l_realinv = 400 * np.diff(np.log(d["realinv"].values))
        gs_l_realgdp = 400 * np.diff(np.log(d["realgdp"].values))
        endogg = gs_l_realinv
        exogg = add_constant(np.c_[gs_l_realgdp, d["realint"][:-1].values])
        res_ols = OLS(endogg, exogg).fit()
        mod_g1 = GLSAR(endogg, exogg, rho=-0.108136)
        res_g1 = mod_g1.fit()
        mod_g2 = GLSAR(endogg, exogg, rho=-0.108136)
        res_g2 = mod_g2.iterative_fit(maxiter=5)
        rho = -0.108136
        partable = np.array(
            [
                [-9.5099, 0.990456, -9.602, 3.65e-18, -11.4631, -7.5567],
                [+4.3704, 0.208146, 21.0, 2.93e-52, 3.95993, 4.78086],
                [-0.579253, 0.268009, -2.161, 0.0319, -1.10777, -0.0507346],
            ]
        )
        result_gretl_g1 = dict(
            endog_mean=("Mean dependent var", 3.113973),
            endog_std=("S.D. dependent var", 18.67447),
            ssr=("Sum squared resid", 22530.9),
            mse_resid_sqrt=("S.E. of regression", 10.66735),
            rsquared=("R-squared", 0.676973),
            rsquared_adj=("Adjusted R-squared", 0.67371),
            fvalue=("F(2, 198)", 221.0475),
            f_pvalue=("P-value(F)", 3.56e-51),
            resid_acf1=("rho", -0.003481),
            dw=("Durbin-Watson", 1.993858),
        )
        reset_2_3 = [5.219019, 0.00619, 2, 197, "f"]
        reset_2 = [7.268492, 0.00762, 1, 198, "f"]
        arch_4 = [7.30776, 0.120491, 4, "chi2"]
        res = res_g1
        assert_almost_equal(res.params, partable[:, 0], 4)
        assert_almost_equal(res.bse, partable[:, 1], 6)
        assert_almost_equal(res.tvalues, partable[:, 2], 2)
        assert_almost_equal(res.ssr, result_gretl_g1["ssr"][1], decimal=2)
        assert_almost_equal(
            np.sqrt(res.mse_resid), result_gretl_g1["mse_resid_sqrt"][1], decimal=5
        )
        assert_almost_equal(res.fvalue, result_gretl_g1["fvalue"][1], decimal=4)
        assert_allclose(res.f_pvalue, result_gretl_g1["f_pvalue"][1], rtol=0.01)
        sm_arch = smsdia.het_arch(res.wresid, nlags=4)
        assert_almost_equal(sm_arch[0], arch_4[0], decimal=4)
        assert_almost_equal(sm_arch[1], arch_4[1], decimal=6)
        res = res_g2
        assert_almost_equal(res.model.rho, rho, decimal=3)
        assert_almost_equal(res.params, partable[:, 0], 4)
        assert_almost_equal(res.bse, partable[:, 1], 3)
        assert_almost_equal(res.tvalues, partable[:, 2], 2)
        assert_almost_equal(res.ssr, result_gretl_g1["ssr"][1], decimal=2)
        assert_almost_equal(
            np.sqrt(res.mse_resid), result_gretl_g1["mse_resid_sqrt"][1], decimal=5
        )
        assert_almost_equal(res.fvalue, result_gretl_g1["fvalue"][1], decimal=0)
        assert_almost_equal(res.f_pvalue, result_gretl_g1["f_pvalue"][1], decimal=6)
        c = oi.reset_ramsey(res, degree=2)
        compare_ftest(c, reset_2, decimal=(2, 4))
        c = oi.reset_ramsey(res, degree=3)
        compare_ftest(c, reset_2_3, decimal=(2, 4))
        sm_arch = smsdia.het_arch(res.wresid, nlags=4)
        assert_almost_equal(sm_arch[0], arch_4[0], decimal=1)
        assert_almost_equal(sm_arch[1], arch_4[1], decimal=2)
        "\n        Performing iterative calculation of rho...\n\n                         ITER       RHO        ESS\n                           1     -0.10734   22530.9\n                           2     -0.10814   22530.9\n\n        Model 4: Cochrane-Orcutt, using observations 1959:3-2009:3 (T = 201)\n        Dependent variable: ds_l_realinv\n        rho = -0.108136\n\n                         coefficient   std. error   t-ratio    p-value\n          -------------------------------------------------------------\n          const           -9.50990      0.990456    -9.602    3.65e-018 ***\n          ds_l_realgdp     4.37040      0.208146    21.00     2.93e-052 ***\n          realint_1       -0.579253     0.268009    -2.161    0.0319    **\n\n        Statistics based on the rho-differenced data:\n\n        Mean dependent var   3.113973   S.D. dependent var   18.67447\n        Sum squared resid    22530.90   S.E. of regression   10.66735\n        R-squared            0.676973   Adjusted R-squared   0.673710\n        F(2, 198)            221.0475   P-value(F)           3.56e-51\n        rho                 -0.003481   Durbin-Watson        1.993858\n        "
        "\n        RESET test for specification (squares and cubes)\n        Test statistic: F = 5.219019,\n        with p-value = P(F(2,197) > 5.21902) = 0.00619\n\n        RESET test for specification (squares only)\n        Test statistic: F = 7.268492,\n        with p-value = P(F(1,198) > 7.26849) = 0.00762\n\n        RESET test for specification (cubes only)\n        Test statistic: F = 5.248951,\n        with p-value = P(F(1,198) > 5.24895) = 0.023:\n        "
        "\n        Test for ARCH of order 4\n\n                     coefficient   std. error   t-ratio   p-value\n          --------------------------------------------------------\n          alpha(0)   97.0386       20.3234       4.775    3.56e-06 ***\n          alpha(1)    0.176114      0.0714698    2.464    0.0146   **\n          alpha(2)   -0.0488339     0.0724981   -0.6736   0.5014\n          alpha(3)   -0.0705413     0.0737058   -0.9571   0.3397\n          alpha(4)    0.0384531     0.0725763    0.5298   0.5968\n\n          Null hypothesis: no ARCH effect is present\n          Test statistic: LM = 7.30776\n          with p-value = P(Chi-square(4) > 7.30776) = 0.120491:\n        "
        "\n        Variance Inflation Factors\n\n        Minimum possible value = 1.0\n        Values > 10.0 may indicate a collinearity problem\n\n           ds_l_realgdp    1.002\n              realint_1    1.002\n\n        VIF(j) = 1/(1 - R(j)^2), where R(j) is the multiple correlation coefficient\n        between variable j and the other independent variables\n\n        Properties of matrix X'X:\n\n         1-norm = 6862.0664\n         Determinant = 1.0296049e+009\n         Reciprocal condition number = 0.013819244\n        "
        "\n        Test for ARCH of order 4 -\n          Null hypothesis: no ARCH effect is present\n          Test statistic: LM = 7.30776\n          with p-value = P(Chi-square(4) > 7.30776) = 0.120491\n\n        Test of common factor restriction -\n          Null hypothesis: restriction is acceptable\n          Test statistic: F(2, 195) = 0.426391\n          with p-value = P(F(2, 195) > 0.426391) = 0.653468\n\n        Test for normality of residual -\n          Null hypothesis: error is normally distributed\n          Test statistic: Chi-square(2) = 20.2792\n          with p-value = 3.94837e-005:\n        "
        "\n        Augmented regression for common factor test\n        OLS, using observations 1959:3-2009:3 (T = 201)\n        Dependent variable: ds_l_realinv\n\n                           coefficient   std. error   t-ratio    p-value\n          ---------------------------------------------------------------\n          const            -10.9481      1.35807      -8.062    7.44e-014 ***\n          ds_l_realgdp       4.28893     0.229459     18.69     2.40e-045 ***\n          realint_1         -0.662644    0.334872     -1.979    0.0492    **\n          ds_l_realinv_1    -0.108892    0.0715042    -1.523    0.1294\n          ds_l_realgdp_1     0.660443    0.390372      1.692    0.0923    *\n          realint_2          0.0769695   0.341527      0.2254   0.8219\n\n          Sum of squared residuals = 22432.8\n\n        Test of common factor restriction\n\n          Test statistic: F(2, 195) = 0.426391, with p-value = 0.653468\n        "
        partable = np.array(
            [
                [-9.48167, 1.17709, -8.055, 7.17e-14, -11.8029, -7.16049],
                [4.37422, 0.328787, 13.3, 2.62e-29, 3.72587, 5.02258],
                [-0.613997, 0.293619, -2.091, 0.0378, -1.193, -0.0349939],
            ]
        )
        result_gretl_g1 = dict(
            endog_mean=("Mean dependent var", 3.257395),
            endog_std=("S.D. dependent var", 18.73915),
            ssr=("Sum squared resid", 22799.68),
            mse_resid_sqrt=("S.E. of regression", 10.7038),
            rsquared=("R-squared", 0.676978),
            rsquared_adj=("Adjusted R-squared", 0.673731),
            fvalue=("F(2, 199)", 90.79971),
            f_pvalue=("P-value(F)", 9.53e-29),
            llf=("Log-likelihood", -763.9752),
            aic=("Akaike criterion", 1533.95),
            bic=("Schwarz criterion", 1543.875),
            hqic=("Hannan-Quinn", 1537.966),
            resid_acf1=("rho", -0.107341),
            dw=("Durbin-Watson", 2.213805),
        )
        linear_squares = [7.52477, 0.0232283, 2, "chi2"]
        arch_4 = [3.43473, 0.487871, 4, "chi2"]
        het_white = [33.503723, 3e-06, 5, "chi2"]
        het_breusch_pagan_konker = [0.709924, 0.7012, 2, "chi2"]
        reset_2_3 = [5.219019, 0.00619, 2, 197, "f"]
        reset_2 = [7.268492, 0.00762, 1, 198, "f"]
        names = "date   residual        leverage       influence        DFFITS".split()
        cur_dir = Path(__file__).parent.resolve()
        fpath = Path(cur_dir).joinpath("results/leverage_influence_ols_nostars.txt")
        lev = np.genfromtxt(
            fpath, skip_header=3, skip_footer=1, converters={0: lambda s: s}
        )
        if np.isnan(lev[-1]["f1"]):
            lev = np.genfromtxt(
                fpath, skip_header=3, skip_footer=2, converters={0: lambda s: s}
            )
        lev.dtype.names = names
        res = res_ols
        cov_hac = sw.cov_hac_simple(res, nlags=4, use_correction=False)
        bse_hac = sw.se_cov(cov_hac)
        assert_almost_equal(res.params, partable[:, 0], 5)
        assert_almost_equal(bse_hac, partable[:, 1], 5)
        assert_almost_equal(res.ssr, result_gretl_g1["ssr"][1], decimal=2)
        assert_almost_equal(res.llf, result_gretl_g1["llf"][1], decimal=4)
        assert_almost_equal(res.rsquared, result_gretl_g1["rsquared"][1], decimal=6)
        assert_almost_equal(
            res.rsquared_adj, result_gretl_g1["rsquared_adj"][1], decimal=6
        )
        assert_almost_equal(
            np.sqrt(res.mse_resid), result_gretl_g1["mse_resid_sqrt"][1], decimal=5
        )
        c = oi.reset_ramsey(res, degree=2)
        compare_ftest(c, reset_2, decimal=(6, 5))
        c = oi.reset_ramsey(res, degree=3)
        compare_ftest(c, reset_2_3, decimal=(6, 5))
        linear_sq = smsdia.linear_lm(res.resid, res.model.exog)
        assert_almost_equal(linear_sq[0], linear_squares[0], decimal=6)
        assert_almost_equal(linear_sq[1], linear_squares[1], decimal=7)
        hbpk = smsdia.het_breuschpagan(res.resid, res.model.exog)
        assert_almost_equal(hbpk[0], het_breusch_pagan_konker[0], decimal=6)
        assert_almost_equal(hbpk[1], het_breusch_pagan_konker[1], decimal=6)
        hw = smsdia.het_white(res.resid, res.model.exog)
        assert_almost_equal(hw[:2], het_white[:2], 6)
        sm_arch = smsdia.het_arch(res.resid, nlags=4)
        assert_almost_equal(sm_arch[0], arch_4[0], decimal=5)
        assert_almost_equal(sm_arch[1], arch_4[1], decimal=6)
        [oi.variance_inflation_factor(res.model.exog, k) for k in [1, 2]]
        infl = oi.OLSInfluence(res_ols)
        assert_almost_equal(lev["residual"], res.resid, decimal=3)
        assert_almost_equal(lev["DFFITS"], infl.dffits[0], decimal=3)
        assert_almost_equal(lev["leverage"], infl.hat_matrix_diag, decimal=3)
        assert_almost_equal(lev["influence"], infl.influence, decimal=4)


def test_GLSARlag():
    from statsmodels.datasets import macrodata

    d2 = macrodata.load_pandas().data
    g_gdp = 400 * np.diff(np.log(d2["realgdp"].values))
    g_inv = 400 * np.diff(np.log(d2["realinv"].values))
    exogg = add_constant(np.c_[g_gdp, d2["realint"][:-1].values], prepend=False)
    mod1 = GLSAR(g_inv, exogg, 1)
    res1 = mod1.iterative_fit(5)
    mod4 = GLSAR(g_inv, exogg, 4)
    res4 = mod4.iterative_fit(10)
    assert_array_less(np.abs(res1.params / res4.params - 1), 0.03)
    assert_array_less(res4.ssr, res1.ssr)
    assert_array_less(np.abs(res4.bse / res1.bse) - 1, 0.015)
    assert_array_less(np.abs((res4.fittedvalues / res1.fittedvalues - 1).mean()), 0.015)
    assert_equal(len(mod4.rho), 4)


if __name__ == "__main__":
    t = TestGLSARGretl()
    t.test_all()
"\nModel 5: OLS, using observations 1959:2-2009:3 (T = 202)\nDependent variable: ds_l_realinv\nHAC standard errors, bandwidth 4 (Bartlett kernel)\n\n                 coefficient   std. error   t-ratio    p-value\n  -------------------------------------------------------------\n  const           -9.48167      1.17709     -8.055    7.17e-014 ***\n  ds_l_realgdp     4.37422      0.328787    13.30     2.62e-029 ***\n  realint_1       -0.613997     0.293619    -2.091    0.0378    **\n\nMean dependent var   3.257395   S.D. dependent var   18.73915\nSum squared resid    22799.68   S.E. of regression   10.70380\nR-squared            0.676978   Adjusted R-squared   0.673731\nF(2, 199)            90.79971   P-value(F)           9.53e-29\nLog-likelihood      -763.9752   Akaike criterion     1533.950\nSchwarz criterion    1543.875   Hannan-Quinn         1537.966\nrho                 -0.107341   Durbin-Watson        2.213805\n\nQLR test for structural break -\n  Null hypothesis: no structural break\n  Test statistic: max F(3, 196) = 3.01985 at observation 2001:4\n  (10 percent critical value = 4.09)\n\nNon-linearity test (logs) -\n  Null hypothesis: relationship is linear\n  Test statistic: LM = 1.68351\n  with p-value = P(Chi-square(2) > 1.68351) = 0.430953\n\nNon-linearity test (squares) -\n  Null hypothesis: relationship is linear\n  Test statistic: LM = 7.52477\n  with p-value = P(Chi-square(2) > 7.52477) = 0.0232283\n\nLM test for autocorrelation up to order 4 -\n  Null hypothesis: no autocorrelation\n  Test statistic: LMF = 1.17928\n  with p-value = P(F(4,195) > 1.17928) = 0.321197\n\nCUSUM test for parameter stability -\n  Null hypothesis: no change in parameters\n  Test statistic: Harvey-Collier t(198) = 0.494432\n  with p-value = P(t(198) > 0.494432) = 0.621549\n\nChow test for structural break at observation 1984:1 -\n  Null hypothesis: no structural break\n  Asymptotic test statistic: Chi-square(3) = 13.1897\n  with p-value = 0.00424384\n\nTest for ARCH of order 4 -\n  Null hypothesis: no ARCH effect is present\n  Test statistic: LM = 3.43473\n  with p-value = P(Chi-square(4) > 3.43473) = 0.487871:\n\n#ANOVA\nAnalysis of Variance:\n\n                     Sum of squares       df      Mean square\n\n  Regression                47782.7        2          23891.3\n  Residual                  22799.7      199          114.571\n  Total                     70582.3      201          351.156\n\n  R^2 = 47782.7 / 70582.3 = 0.676978\n  F(2, 199) = 23891.3 / 114.571 = 208.528 [p-value 1.47e-049]\n\n#LM-test autocorrelation\nBreusch-Godfrey test for autocorrelation up to order 4\nOLS, using observations 1959:2-2009:3 (T = 202)\nDependent variable: uhat\n\n                 coefficient   std. error   t-ratio    p-value\n  ------------------------------------------------------------\n  const           0.0640964    1.06719       0.06006   0.9522\n  ds_l_realgdp   -0.0456010    0.217377     -0.2098    0.8341\n  realint_1       0.0511769    0.293136      0.1746    0.8616\n  uhat_1         -0.104707     0.0719948    -1.454     0.1475\n  uhat_2         -0.00898483   0.0742817    -0.1210    0.9039\n  uhat_3          0.0837332    0.0735015     1.139     0.2560\n  uhat_4         -0.0636242    0.0737363    -0.8629    0.3893\n\n  Unadjusted R-squared = 0.023619\n\nTest statistic: LMF = 1.179281,\nwith p-value = P(F(4,195) > 1.17928) = 0.321\n\nAlternative statistic: TR^2 = 4.771043,\nwith p-value = P(Chi-square(4) > 4.77104) = 0.312\n\nLjung-Box Q' = 5.23587,\nwith p-value = P(Chi-square(4) > 5.23587) = 0.264:\n\nRESET test for specification (squares and cubes)\nTest statistic: F = 5.219019,\nwith p-value = P(F(2,197) > 5.21902) = 0.00619\n\nRESET test for specification (squares only)\nTest statistic: F = 7.268492,\nwith p-value = P(F(1,198) > 7.26849) = 0.00762\n\nRESET test for specification (cubes only)\nTest statistic: F = 5.248951,\nwith p-value = P(F(1,198) > 5.24895) = 0.023\n\n#heteroscedasticity White\nWhite's test for heteroskedasticity\nOLS, using observations 1959:2-2009:3 (T = 202)\nDependent variable: uhat^2\n\n                  coefficient   std. error   t-ratio   p-value\n  -------------------------------------------------------------\n  const           104.920       21.5848       4.861    2.39e-06 ***\n  ds_l_realgdp    -29.7040       6.24983     -4.753    3.88e-06 ***\n  realint_1        -6.93102      6.95607     -0.9964   0.3203\n  sq_ds_l_realg     4.12054      0.684920     6.016    8.62e-09 ***\n  X2_X3             2.89685      1.38571      2.091    0.0379   **\n  sq_realint_1      0.662135     1.10919      0.5970   0.5512\n\n  Unadjusted R-squared = 0.165860\n\nTest statistic: TR^2 = 33.503723,\nwith p-value = P(Chi-square(5) > 33.503723) = 0.000003:\n\n#heteroscedasticity Breusch-Pagan (original)\nBreusch-Pagan test for heteroskedasticity\nOLS, using observations 1959:2-2009:3 (T = 202)\nDependent variable: scaled uhat^2\n\n                 coefficient   std. error   t-ratio    p-value\n  -------------------------------------------------------------\n  const           1.09468      0.192281      5.693     4.43e-08 ***\n  ds_l_realgdp   -0.0323119    0.0386353    -0.8363    0.4040\n  realint_1       0.00410778   0.0512274     0.08019   0.9362\n\n  Explained sum of squares = 2.60403\n\nTest statistic: LM = 1.302014,\nwith p-value = P(Chi-square(2) > 1.302014) = 0.521520\n\n#heteroscedasticity Breusch-Pagan Koenker\nBreusch-Pagan test for heteroskedasticity\nOLS, using observations 1959:2-2009:3 (T = 202)\nDependent variable: scaled uhat^2 (Koenker robust variant)\n\n                 coefficient   std. error   t-ratio    p-value\n  ------------------------------------------------------------\n  const           10.6870       21.7027      0.4924    0.6230\n  ds_l_realgdp    -3.64704       4.36075    -0.8363    0.4040\n  realint_1        0.463643      5.78202     0.08019   0.9362\n\n  Explained sum of squares = 33174.2\n\nTest statistic: LM = 0.709924,\nwith p-value = P(Chi-square(2) > 0.709924) = 0.701200\n\n########## forecast\n#forecast mean y\n For 95% confidence intervals, t(199, 0.025) = 1.972\n\n     Obs ds_l_realinv    prediction    std. error        95% interval\n\n  2008:3     -7.134492   -17.177905     2.946312   -22.987904 - -11.367905\n  2008:4    -27.665860   -36.294434     3.036851   -42.282972 - -30.305896\n  2009:1    -70.239280   -44.018178     4.007017   -51.919841 - -36.116516\n  2009:2    -27.024588   -12.284842     1.427414   -15.099640 - -9.470044\n  2009:3      8.078897     4.483669     1.315876     1.888819 - 7.078520\n\n  Forecast evaluation statistics\n\n  Mean Error                       -3.7387\n  Mean Squared Error                218.61\n  Root Mean Squared Error           14.785\n  Mean Absolute Error               12.646\n  Mean Percentage Error            -7.1173\n  Mean Absolute Percentage Error   -43.867\n  Theil's U                         0.4365\n  Bias proportion, UM               0.06394\n  Regression proportion, UR         0.13557\n  Disturbance proportion, UD        0.80049\n\n#forecast actual y\n For 95% confidence intervals, t(199, 0.025) = 1.972\n\n     Obs ds_l_realinv    prediction    std. error        95% interval\n\n  2008:3     -7.134492   -17.177905    11.101892   -39.070353 - 4.714544\n  2008:4    -27.665860   -36.294434    11.126262   -58.234939 - -14.353928\n  2009:1    -70.239280   -44.018178    11.429236   -66.556135 - -21.480222\n  2009:2    -27.024588   -12.284842    10.798554   -33.579120 - 9.009436\n  2009:3      8.078897     4.483669    10.784377   -16.782652 - 25.749991\n\n  Forecast evaluation statistics\n\n  Mean Error                       -3.7387\n  Mean Squared Error                218.61\n  Root Mean Squared Error           14.785\n  Mean Absolute Error               12.646\n  Mean Percentage Error            -7.1173\n  Mean Absolute Percentage Error   -43.867\n  Theil's U                         0.4365\n  Bias proportion, UM               0.06394\n  Regression proportion, UR         0.13557\n  Disturbance proportion, UD        0.80049\n\n"
