# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 10:01:25 2011

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from scikits.statsmodels.iolib import SimpleTable
from scikits.statsmodels.iolib.summary import summary_params, forg
from scikits.statsmodels.iolib.tableformatting import fmt_params



#from scikits.statsmodels.iolib.summary import Summary
#smry = Summary()
#smry.add_table_2cols(self, gleft=top_left, gright=top_right,
#                  yname=yname, xname=xname, title=title)
#smry.add_table_params(self, yname=yname, xname=xname, alpha=.05,
#                     use_t=True)