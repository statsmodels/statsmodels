
import numpy as np

try:
    from numpy.linalg import slogdet as np_slogdet
except:
    def np_slogdet(x):
        return 1, np.log(np.linalg.det(x))
