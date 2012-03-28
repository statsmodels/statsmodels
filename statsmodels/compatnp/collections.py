'''backported compatibility functions for Python's collections

'''

try:
    #python >= 2.7
    from collections import OrderedDict
except ImportError:
    #http://code.activestate.com/recipes/576693/
    #author: Raymond Hettinger
    from ordereddict import OrderedDict
