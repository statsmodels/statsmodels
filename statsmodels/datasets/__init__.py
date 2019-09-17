"""
Datasets module
"""
from statsmodels.tools._testing import PytestTester

from . import (anes96, cancer, committee, ccard, copper, cpunish, elnino,
               engel, grunfeld, interest_inflation, longley, macrodata,
               modechoice, nile, randhie, scotland, spector, stackloss,
               star98, strikes, sunspots, fair, heart, statecrime, co2,
               fertility, china_smoking)
from .utils import (get_rdataset, get_data_home, clear_data_home, webuse,
                    check_internet)

__all__ = ['anes96', 'cancer', 'committee', 'ccard', 'copper', 'cpunish',
           'elnino', 'engel', 'grunfeld', 'interest_inflation', 'longley',
           'macrodata', 'modechoice', 'nile', 'randhie', 'scotland', 'spector',
           'stackloss', 'star98', 'strikes', 'sunspots', 'fair', 'heart',
           'statecrime', 'co2', 'fertility', 'china_smoking',
           'get_rdataset', 'get_data_home', 'clear_data_home', 'webuse',
           'check_internet', 'test']

test = PytestTester()
