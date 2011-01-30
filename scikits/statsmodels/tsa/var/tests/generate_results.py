from rpy2.robjects import r
import pandas as pn
import pandas.rpy.common as prp
from scikits.statsmodels.tsa.var.model import VAR

basepath = 'scikits/statsmodels/tsa/var/tests/'

def generate_var():
    r.source(basepath + 'var.R')
    return prp.convert_robj(r['result'])

if __name__ == '__main__':
    pass
