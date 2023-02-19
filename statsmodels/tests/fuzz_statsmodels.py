#!/usr/bin/python3

import atheris
import sys

with atheris.instrument_imports():
    import statsmodels.api as sm
    from statsmodels.multivariate.pca import PCA
    from statsmodels.tsa.forecasting.theta import ThetaModel
    import statsmodels.stats.libqsturng.qsturng_ as qs
    from statsmodels.iolib.table import SimpleTable
    import numpy as np
    import warnings


# creates an array
def ConsumeListofLists(fdp, num_lists):
    l = []
    for _ in range(num_lists):
        l.append(fdp.ConsumeIntListInRange(num_lists, 1, 1000))
    return l


def TestOneInput(data):
    warnings.filterwarnings("ignore")
    if len(data) == 0:
        return
    fdp = atheris.FuzzedDataProvider(data)

    # data to fuzz test with
    x = fdp.ConsumeIntListInRange(10, 1, 100)
    y = fdp.ConsumeIntListInRange(10, 1, 100)

    # creates an array of varying sizes
    num_lists = fdp.ConsumeIntInRange(1, 100)
    b = np.array(ConsumeListofLists(fdp, num_lists))

    # fuzz SimpleTable
    fuzz_data = ConsumeListofLists(fdp, num_lists)
    SimpleTable(fuzz_data)

    # fuzz PCA
    PCA(b, standardize=False, demean=False, normalize=False)

    # fuzz OLS
    ols_model = sm.OLS(y, x)
    ols_model.fit()

    # fuzz GLS
    gls_model = sm.GLS(y, x)
    gls_model.fit()

    # fuzz WLS
    wls_model = sm.WLS(y, x)
    wls_model.fit()

    # fuzz GLMs
    # keep getting "ValueError: The first guess on the deviance
    # function returned a nan. This could be a boundary problem
    # and should be reported."
    # Ignoring it for now
    try:
        bi_list = fdp.ConsumeIntListInRange(10, 0, 1)
        glm_model1 = sm.GLM(bi_list, x, family=sm.families.Binomial())
        glm_model1.fit()
        glm_model2 = sm.GLM(y, x, family=sm.families.Gamma())
        glm_model2.fit()
        glm_model3 = sm.GLM(y, x, family=sm.families.Gaussian())
        glm_model3.fit()
    except ValueError:
        pass

    # fuzz ThetaModel
    z = np.array(y)
    tm_model = ThetaModel(z, period=len(y))
    tm_model.fit()

    # fuzz psturng and _qsturng
    qs.psturng(fdp.ConsumeIntInRange(1, 1000),
               fdp.ConsumeIntInRange(2, 1000),
               fdp.ConsumeIntInRange(2, 1000))
    qs._qsturng(np.array(fdp.ConsumeFloatInRange(0.1, 0.999)),
                np.array(fdp.ConsumeIntInRange(2, 1000)),
                fdp.ConsumeIntInRange(2, 1000))


atheris.Setup(sys.argv, TestOneInput)
atheris.Fuzz()
