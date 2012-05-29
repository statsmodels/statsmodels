'''
Hard-coded results for test_sysreg
'''

import numpy as np

class GrunfeldSUR(object):
   '''
   The results for the Grunfeld dataset were obtained from systemfit.
   For more details see sysreg/tests/results/grunfeld-sur.R
   '''
   def __init__(self):
      self.params =np.array([0.9979992,0.06886083,0.3083878,-21.1374,
         0.03705313,0.1286866,-168.1134,0.1219063,0.3821666,62.25631,
         0.1214024,0.3691114,1.407487,0.05635611,0.04290209]).reshape((15,1))


