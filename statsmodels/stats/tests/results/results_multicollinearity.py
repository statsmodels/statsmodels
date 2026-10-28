"""
Reference values for statsmodels.stats.multicollinearity

The values were computed with R 4.6.1 and the car package, version 3.1.5,
using the following code. The data frames ``longley`` and ``mtcars`` are part
of the datasets package that comes with R. The Longley regressors,
GNP.deflator, GNP, Unemployed, Armed.Forces, Population, Year, are in the same
order as the exog of ``statsmodels.datasets.longley`` (GNPDEFL, GNP, UNEMP,
ARMED, POP, YEAR). The units of some variables differ between the two, for
example GNP, which does not matter for the VIF.

``car::vif`` returns the diagonal of the inverse of the correlation matrix of
the regressors without the constant (generalized VIF with one degree of
freedom per term), which is the VIF of the regressions with a constant.

.. code-block:: R

    library(car)
    # 1. Longley, six regressors and a subset of four regressors
    fit <- lm(Employed ~ GNP.deflator + GNP + Unemployed + Armed.Forces +
                Population + Year, data = longley)
    vif(fit)
    fit2 <- lm(Employed ~ GNP + Unemployed + Armed.Forces + Population,
               data = longley)
    vif(fit2)

    # 2. mtcars
    fit3 <- lm(mpg ~ disp + hp + wt + drat + qsec, data = mtcars)
    vif(fit3)
    dput(mtcars$disp)  # and hp, wt, drat, qsec

    # 3. Exactly collinear designs. The coefficients of the aliased columns
    # of lm are NA. The later column of a collinear relation is aliased.
    L <- longley
    X1 <- cbind(const = 1, GNP = L$GNP, Unemployed = L$Unemployed,
                GNP_plus_Unemployed = L$GNP + L$Unemployed,
                Armed.Forces = L$Armed.Forces, Population = L$Population)
    is.na(coef(lm(L$Employed ~ 0 + X1)))
    # constant and a full set of dummy variables for Year %% 4
    D <- model.matrix(~ 0 + factor(L$Year %% 4))
    X2 <- cbind(const = 1, GNP = L$GNP, D, Armed.Forces = L$Armed.Forces)
    is.na(coef(lm(L$Employed ~ 0 + X2)))
"""

import numpy as np

# car::vif for the regressors GNPDEFL, GNP, UNEMP, ARMED, POP, YEAR
longley_vif = np.array(
    [
        135.5324382799854277,
        1788.5134827179381318,
        33.6188905960495745,
        3.5889301934453179,
        399.1510223125925449,
        758.9805974067925263,
    ]
)

# car::vif for the regressors GNP, UNEMP, ARMED, POP
longley4_vif = np.array(
    [
        175.7031445330708834,
        5.1730786516670060,
        2.2799081477059446,
        196.5953072082543258,
    ]
)

# fmt: off
mtcars = {
    "disp": [
        160, 160, 108, 258, 360, 225, 360, 146.7, 140.8, 167.6, 167.6,
        275.8, 275.8, 275.8, 472, 460, 440, 78.7, 75.7, 71.1, 120.1,
        318, 304, 350, 400, 79, 120.3, 95.1, 351, 145, 301, 121,
    ],
    "hp": [
        110, 110, 93, 110, 175, 105, 245, 62, 95, 123, 123, 180, 180,
        180, 205, 215, 230, 66, 52, 65, 97, 150, 150, 245, 175, 66, 91,
        113, 264, 175, 335, 109,
    ],
    "wt": [
        2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44,
        3.44, 4.07, 3.73, 3.78, 5.25, 5.424, 5.345, 2.2, 1.615, 1.835,
        2.465, 3.52, 3.435, 3.84, 3.845, 1.935, 2.14, 1.513, 3.17, 2.77,
        3.57, 2.78,
    ],
    "drat": [
        3.9, 3.9, 3.85, 3.08, 3.15, 2.76, 3.21, 3.69, 3.92, 3.92, 3.92,
        3.07, 3.07, 3.07, 2.93, 3, 3.23, 4.08, 4.93, 4.22, 3.7, 2.76,
        3.15, 3.73, 3.08, 4.08, 4.43, 3.77, 4.22, 3.62, 3.54, 4.11,
    ],
    "qsec": [
        16.46, 17.02, 18.61, 19.44, 17.02, 20.22, 15.84, 20, 22.9,
        18.3, 18.9, 17.4, 17.6, 18, 17.98, 17.82, 17.42, 19.47, 18.52,
        19.9, 20.01, 16.87, 17.3, 15.41, 17.05, 18.9, 16.7, 16.9, 14.5,
        15.5, 14.6, 18.6,
    ],
}
# fmt: on

# car::vif for the regressors disp, hp, wt, drat, qsec
mtcars_vif = np.array(
    [
        9.1108693142086477,
        5.2018330126530898,
        7.0126856976392711,
        2.3223426460928769,
        3.1919392872710808,
    ]
)

# Position (0-based) of the columns that are aliased in lm, i.e. that are
# collinear with preceding columns, for the two designs of the R code.
# const, GNP, UNEMP, GNP + UNEMP, ARMED, POP
alias_sum = [3]
# const, GNP, four dummy variables, ARMED
alias_dummies = [5]
