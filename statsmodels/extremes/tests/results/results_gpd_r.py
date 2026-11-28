"""
Reference values from R for GeneralizedParetoModel.

Data: gpd_covariates.csv in this directory (600 simulated exceedances).

Produced with R 4.3.3, ismev 1.43 and extRemes 2.2-1 by this script, run
in the directory that holds gpd_covariates.csv:

    # Reference values for statsmodels GeneralizedParetoModel.
    # Run with: Rscript gpd_reference.R   (needs install.packages(c("ismev", "extRemes")))
    # Uses only R's own fitting functions on the exported data. No code is ported.
    options(digits = 12)
    d <- read.csv("gpd_covariates.csv")
    ydat <- as.matrix(d[, c("x1", "x2")])

    fit_ismev <- function(shl) {
      # Exceedances are y itself (threshold 0). sigma = exp(b0 + b1 x1 + b2 x2).
      # Parameter order in $mle: sigma coefficients, then shape coefficients.
      ismev::gpd.fit(d$y, threshold = 0, ydat = ydat, sigl = c(1, 2), shl = shl,
                     siglink = exp,
                     siginit = c(log(mean(d$y)), 0, 0),
                     shinit = if (is.null(shl)) 0.1 else c(0.1, 0),
                     method = "BFGS", maxit = 100000, reltol = 1e-14, show = FALSE)
    }

    for (shl in list(NULL, 2)) {
      f <- fit_ismev(shl)
      cat(if (is.null(shl)) "ismev, shape constant\n" else "ismev, shape on x2\n")
      cat("  mle ", format(f$mle, digits = 12), "\n")
      cat("  se  ", format(f$se, digits = 12), "\n")
      cat("  nllh", format(f$nllh, digits = 15), "\n")
    }

    # Second, independent implementation. use.phi = TRUE puts the scale on the log scale.
    f2 <- extRemes::fevd(y, data = d, threshold = 0, type = "GP",
                         scale.fun = ~ x1 + x2, shape.fun = ~ x2, use.phi = TRUE)
    cat("extRemes, shape on x2\n")
    print(summary(f2))
"""

import numpy as np

from statsmodels.tools.testing import Holder

# ismev::gpd.fit, log(scale) on x1 and x2, shape constant
ismev_shape_const = Holder()
ismev_shape_const.params = np.array(
    [0.2027237448078, 0.5103208539411, -0.0718438400453, 0.1611561870904]
)
ismev_shape_const.bse = np.array(
    [0.1007727487732, 0.0469693846584, 0.1638593542285, 0.0483621236331]
)
ismev_shape_const.nllh = 803.193987196091

# ismev::gpd.fit, log(scale) on x1 and x2, shape on x2
ismev_shape_x2 = Holder()
ismev_shape_x2.params = np.array(
    [
        0.2726114572232,
        0.5143515322039,
        -0.2229854353836,
        0.0821610067988,
        0.1674576317816,
    ]
)
ismev_shape_x2.bse = np.array(
    [
        0.1228375550224,
        0.0472647678022,
        0.2259428473054,
        0.0889120390142,
        0.1704226019288,
    ]
)
ismev_shape_x2.nllh = 802.727445706197

# extRemes::fevd, use.phi = TRUE, log(scale) on x1 and x2, shape on x2
extremes_shape_x2 = Holder()
extremes_shape_x2.params = np.array(
    [
        0.2726214993335,
        0.5143391014221,
        -0.2230333099520,
        0.0821301428573,
        0.1675309512592,
    ]
)
extremes_shape_x2.bse = np.array(
    [
        0.1228364114029,
        0.0472647003720,
        0.2259420055476,
        0.0889052451866,
        0.1704177690310,
    ]
)
extremes_shape_x2.nllh = 802.727445858
