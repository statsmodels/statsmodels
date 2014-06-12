import numpy as np
import pandas as pd
from statsmodels.sandbox.mice import mice

def test_get_exog():

    np.random.seed(1325)
    data = np.random.normal(size=(10,4))
    data[8:, 1] = np.inf
    df = pd.DataFrame(data, columns=["X1", "X2", "X3", "X4"])

    imp_dat = mice.ImputedData(df)
    ex = imp_dat.get_exog("X3+X4", "X2", select="observed")
    ex = np.asarray(ex)
    np.assert_almost_equal(ex, data[0:8, [3,4]])

if __name__=="__main__":
    import nose
    nose.runmodule(argv=[__file__,'-vvs','-x',#'--pdb', '--pdb-failure'
                         ],
                   exit=False)    