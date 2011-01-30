def generate_var():
    from rpy2.robjects import r
    import pandas.rpy.common as prp
    r.source('tests/var.R')
    return prp.convert_robj(r['result'], use_pandas=False)

if __name__ == '__main__':
    import numpy as np
    result = generate_var()
    np.savez('tests/results/vars_results.npz', **result)
