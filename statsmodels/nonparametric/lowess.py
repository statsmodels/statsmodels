import warnings
warnings.warn('the module lowess is deprecated and will be removed in 0.5.'
              'The new module name is smoothers_lowess, and lowess (function)'
              'can be imported now through statsmodels.api')
from smoothers_lowess import *