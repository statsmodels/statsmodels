"""
Datasets module
"""
from . import (anes96, cancer, committee, ccard, copper, cpunish,  # noqa:F401
               elnino, engel, grunfeld, interest_inflation, longley,
               macrodata, modechoice, nile, randhie, scotland, spector,
               stackloss,  star98, strikes, sunspots, fair, heart,
               statecrime, co2, fertility, china_smoking)
from .utils import (get_rdataset, get_data_home, clear_data_home,  # noqa:F401
                    webuse, check_internet)
