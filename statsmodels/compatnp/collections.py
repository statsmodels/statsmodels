'''backported compatibility functions for Python's collections

'''

try:
    #python >= 2.7
    from collections import OrderedDict
except ImportError:
    #http://code.activestate.com/recipes/576693/
    #author: Raymond Hettinger
    from .ordereddict import OrderedDict

try:
    #python >= 2.7
    from collections import Counter
except ImportError:
    #http://code.activestate.com/recipes/576611/
    #author: Raymond Hettinger
    from .counter import Counter
