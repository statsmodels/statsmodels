"""
Poisson counts with a large exog, to test the step halving of Newton's method
and of IRLS, shared by the tests of GLM and of the discrete models.

The references are R 4.6.1 with

    fit <- glm(y ~ x, family = poisson(),
               control = glm.control(epsilon = 1e-14, maxit = 100))
    coef(fit); sqrt(diag(vcov(fit))); as.numeric(logLik(fit))

with ``x = seq(50, 100, length.out = n)``. The tight ``epsilon`` converges the
fit to the maximum of the log-likelihood. With the default ``epsilon`` the
standard errors of the first data set differ from these by a relative 1.3e-6
and the parameters by 4e-12.

``large``: n = 25. The first full step of Newton's method or IRLS from zeros
has a linear predictor of at most 96, so that ``exp`` does not overflow, but
the full steps need 96 iterations (36 from ``[1, 0]``). With step halving they
need 7 or less.

``overflow``: n = 40. The first full step from zeros has a linear predictor of
1539, so that ``exp`` overflows, and Newton's method returns nan parameters
and IRLS raises an error without step halving. ``glm`` in R with
``start = c(0, 0)`` fails with "NA/NaN/Inf in 'x'" for these data, and only
its default starting values work.
"""

import numpy as np

x_large = np.linspace(50, 100, 25)
y_large = np.array(
    [1, 0, 1, 6, 6, 5, 6, 3, 12, 11, 8, 11, 10, 22, 19, 25, 43, 39, 51, 67, 66,
     97, 107, 113, 165]
)
params_large = [-4.0750125697522002, 0.091257238998032855]
bse_large = [0.31088227488861059, 0.0034150824900276108]
llf_large = -69.710685444885669

x_overflow = np.linspace(50, 100, 40)
y_overflow = np.array(
    [5, 10, 11, 9, 25, 14, 20, 26, 28, 21, 45, 43, 52, 48, 67, 83, 73, 98, 105,
     130, 152, 183, 204, 266, 300, 328, 411, 490, 540, 615, 780, 853, 1035,
     1189, 1299, 1596, 1900, 2194, 2507, 2909]
)
params_overflow = [-3.9525979059627634, 0.11938444519416855]
bse_overflow = [0.080470355764021881, 0.00086799628559341589]
llf_overflow = -155.05237286776074
