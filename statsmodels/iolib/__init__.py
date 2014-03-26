from .foreign import StataReader, genfromdta, savetxt
from .table import SimpleTable, csv2st
from .smpickle import save_pickle, load_pickle

from statsmodels import NoseWrapper as Tester
test = Tester().test
