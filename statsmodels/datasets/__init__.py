"""
Datasets module
"""
#__all__ = filter(lambda s:not s.startswith('_'),dir())
from . import (anes96, birthwt, cancer, committee, ccard, copper, cpunish, elnino,
               engel, grunfeld, longley, macrodata, modechoice, nile, randhie,
               scotland, spector, stackloss, star98, strikes, sunspots, fair,
               heart, statecrime, co2, wage1)
from .utils import get_rdataset, get_data_home, clear_data_home
