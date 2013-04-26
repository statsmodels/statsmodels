"""
Datasets module
"""
#__all__ = filter(lambda s:not s.startswith('_'),dir())
from . import (anes96, cancer, committee, ccard, copper, cpunish, elnino,
               engel, grunfeld, longley, macrodata, nile, randhie, scotland,
               spector, stackloss, star98, strikes, sunspots, fair, heart,
               statecrime)
from utils import get_rdataset, get_data_home, clear_data_home
