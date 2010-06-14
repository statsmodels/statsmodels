from scipy.stats import norm
from numpy import array, polyval
import numpy as np #TODO: remove

# These are the cut-off values for the left-tail vs. the rest of the
# distribution

z_star_nc = [-2.9,-8.7,-14.8,-20.9,-25.7,-30.4]
z_star_c = [-8.9,-14.3,-19.5,-25.1,-29.6,-34.4]
z_star_ct = [-15.0,-19.6,-25.3,-29.6,-31.8,-38.4]
z_star_ctt = [-20.7,-25.3,-29.9,-34.4,-38.5,-44.2]


# These are Table 5 from MacKinnon (1994)
# small p is defined as p in .005 to .150 ie p = .005 up to z_star
# Z* is the largest value for which it is appropriate to use these
# approximations
# the left tail approximation is
# p = norm.cdf(d_0 + d_1*log(abs(z)) + d_2*log(abs(z))**2 + d_3*log(abs(z))**3
# there is no Z-min, ie., it is well-behaved in the left tail

znc_smallp = array([[.0342, -.6376,0,-.03872],
                    [1.3426,-.7680,0,-.04104],
                    [3.8607,-2.4159,.51293,-.09835],
                    [6.1072,-3.7250,.85887,-.13102],
                    [7.7800,-4.4579,1.00056,-.14014],
                    [4.0253, -.8815,0,-.04887]])

zc_smallp = array([[2.2142,-1.7863,.32828,-.07727],
                   [1.1662,.1814,-.36707,0],
                   [6.6584,-4.3486,1.04705,-.15011],
                   [3.3249,-.8456,0,-.04818],
                   [4.0356,-.9306,0,-.04776],
                   [13.9959,-8.4314,1.97411,-.2234]])

# These are Table 6 from MacKinnon (1994), note that the last columns are
# the same as the last columns of Table 5.  These are well-behaved in the
# right tail.
# the approximation function is
# p = norm.cdf(d_0 + d_1 * z + d_2*z**2 + d_3*z**3 + d_4*z**4)
scaling = [1,10e-2,10e-3,10e-4,10e-6]
znc_largep = array([[.4927,6.9060,13.2331,12.0990,0],
                    [1.5167,4.6859,4.2401,2.7939,7.9601],
                    [2.2347,3.9465,2.2406,.8746,1.4239],
                    [2.8239,3.6265,1.6738,.5408,.7449],
                    [3.3174,3.3492,1.2792,.3416,.3894],
                    [3.7290,3.0611,.9579,.2087,.1943]])
zc_largep = array([[1.7170,5.5243,4.3463,1.6671,0],
                   [2.2394,4.2377,2.4320,.9241,.4364],
                   [2.7430,3.6260,1.5703,.4612,.5670],
                   [3.2280,3.3399,1.2319,.3162,.6482],
                   [3.6583,3.0934,.9681,.2111,.1979],
                   [4.0379,2.8735,.7694,.1433,.1146]])

znc_largep *= scaling
zc_largep *= scaling

# For P is small
def mackinnont14(tau):
    return norm.cdf(1.8951 + 1.2236*tau)

def mackinnont15(tau):
    return norm.cdf(1.7760 + 1.0448*tau - .0578*tau**2)

def mackinnont16(tau):
    return norm.cdf(1.7325 + .8898*tau - .1836*tau**2 - .0282*tau**3)

def mackinnon_z_p_adf(teststat, regression="c", k=1):
    """
    Returns MacKinnon's approximate p-value for Z(teststat).

    Parameters
    ----------
    teststat : float
        "T-value" from an Augmented Dickey-Fuller regression.
    regression : str {"c", "nc", "ct", "ctt"}
        This is the method of regression that was used.  Following MacKinnon's
        notation, this can be "c" for constant, "nc" for no constant, "ct" for
        constant and trend, and "ctt" for constant, trend, and trend-squared.

    References
    ----------
    MacKinnon, J.G. 1994  "Approximate Asymptotic Distribution Functions for
        Unit-Root and Cointegration Tests." Journal of Business & Economics
        Statistics, 12.2, 167-76.
    """
    if regression=="c":
        if teststat > z_star_c[k-1]:    # does k=0 for constant only?
            z_coef = zc_largep[k-1]
        else:
            z_coef = zc_smallp[k-1]
            teststat = np.log(np.abs(teststat))
        return norm.cdf(polyval(z_coef[::-1], teststat))




if __name__=="__main__":
    tau = float(raw_input("What is tau? "))
# for tau it is just t-stat
    tau = -1.2009658
# for z statistic it is
    nobs = 91 # for Lutkepohl2; dfuller ln_inv in Stata
    coef = -.0121366
    z = nobs*coef
    print mackinnont14(tau)
    print mackinnont15(tau)
    print mackinnont16(tau)

    print mackinnon_z_p_adf(z)
