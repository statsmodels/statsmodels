from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keyword import iskeyword as _iskeyword
from operator import itemgetter as _itemgetter
from collections import OrderedDict
import sys

from ..compat.python import string_types


def namedtuple(typename, field_names, verbose=False, rename=False, doc="",
               field_docs=None):
    """Returns a new subclass of tuple with named fields.

    >>> Point = namedtuple('Point', 'x y')
    # docstring for the new class
    >>> Point.__doc__
    'Point(x, y)'
    # instantiate with positional args or keywords
    >>> p = Point(11, y=22)
    # indexable like a plain tuple
    >>> p[0] + p[1]
    33
    # unpack like a regular tuple
    >>> x, y = p
    >>> x, y
    (11, 22)
    # fields also accessible by name
    >>> p.x + p.y
    33
    # convert to a dictionary
    >>> d = p._asdict()
    >>> d['x']
    11
    # convert from a dictionary
    >>> Point(**d)
    Point(x=11, y=22)
    # _replace() is like str.replace() but targets named fields
    >>> p._replace(x=100)
    Point(x=100, y=22)

    """

    # Parse and validate the field names.  Validation serves two purposes,
    # generating informative error messages and preventing template injection
    # attacks.
    if isinstance(field_names, string_types):
        # names separated by whitespace and/or commas
        field_names = field_names.replace(',', ' ').split()
    field_names = tuple(map(str, field_names))
    forbidden_fields = set(['__init__', '__slots__', '__new__', '__repr__',
                            '__getnewargs__'])
    if rename:
        names = list(field_names)
        seen = set()
        for i, name in enumerate(names):
            need_suffix = (not all(c.isalnum() or c == '_' for c in name)
                           or _iskeyword(name) or not name or name[0].isdigit()
                           or name.startswith('_') or name in seen)
            if need_suffix:
                names[i] = '_%d' % i
            seen.add(name)
        field_names = tuple(names)
    for name in (typename,) + field_names:
        if not all(c.isalnum() or c == '_' for c in name):
            raise ValueError('Type names and field names can only contain '
                             'alphanumeric characters and underscores: %r'
                             % name)
        if _iskeyword(name):
            raise ValueError('Type names and field names cannot be a keyword: '
                             '%r' % name)
        if name[0].isdigit():
            raise ValueError('Type names and field names cannot start with a '
                             'number: %r' % name)
    seen_names = set()
    for name in field_names:
        if name.startswith('__'):
            if name in forbidden_fields:
                raise ValueError('Field names cannot be on of %s'
                                 % ', '.join(forbidden_fields))
        elif name.startswith('_') and not rename:
            raise ValueError('Field names cannot start with an underscore: %r'
                             % name)
        if name in seen_names:
            raise ValueError('Encountered duplicate field name: %r' % name)
        seen_names.add(name)

    # Create and fill-in the class template
    numfields = len(field_names)
    # tuple repr without parens or quotes
    argtxt = repr(field_names).replace("'", "")[1:-1]
    reprtxt = ', '.join('%s=%%r' % name for name in field_names)
    if doc is None:
        doc = '%(typename)s(%(argtxt)s)' % dict(typename=typename,
                                                argtxt=argtxt)
    template = '''class %(typename)s(tuple):
        %(doc)s\n
        __slots__ = () \n
        _fields = %(field_names)r \n
        def __new__(_cls, %(argtxt)s):
            'Create new instance of %(typename)s(%(argtxt)s)'
            return _tuple.__new__(_cls, (%(argtxt)s)) \n
        @classmethod
        def _make(cls, iterable, new=tuple.__new__, len=len):
            'Make a new %(typename)s object from a sequence or iterable'
            result = new(cls, iterable)
            if len(result) != %(numfields)d:
                raise TypeError('Expected %(numfields)d arguments, got %%d'
                                %% len(result))
            return result \n
        def __repr__(self):
            'Return a nicely formatted representation string'
            return '%(typename)s(%(reprtxt)s)' %% self \n
        def _asdict(self):
            'Return a new OrderedDict which maps field names to their values'
            return OrderedDict(zip(self._fields, self)) \n
        def _replace(_self, **kwds):
            'Return a new %(typename)s object replacing specified values'
            result = _self._make(map(kwds.pop, %(field_names)r, _self))
            if kwds:
                raise ValueError('Got unexpected field names: %%r'
                                 %% kwds.keys())
            return result \n
        def __getnewargs__(self):
            'Return self as a plain tuple.  Used by copy and pickle.'
            return tuple(self) \n\n''' % dict(
            numfields=numfields, field_names=field_names, typename=typename,
            argtxt=argtxt, reprtxt=reprtxt, doc=repr(doc)
        )
    if field_docs is None:
        field_docs = ['Alias for field number %d' % i
                      for i in range(len(field_names))]
    for i, name in enumerate(field_names):
        template += "        %s = _property(_itemgetter(%d), " \
                    "doc=%s)\n" % (name, i, repr(field_docs[i]))
    if verbose:
        print(template)

    # Execute the template string in a temporary namespace and support tracing
    # utilities by setting a value for frame.f_globals['__name__']
    namespace = dict(
        _itemgetter=_itemgetter, __name__='namedtuple_%s' % typename,
        OrderedDict=OrderedDict, _property=property, _tuple=tuple
    )
    try:
        exec(template, namespace)
    except SyntaxError as e:
        raise SyntaxError(e.message + ':\n' + template)
    result = namespace[typename]

    # For pickling to work, the __module__ variable needs to be set to the
    # frame where the named tuple is created.  Bypass this step in environments
    # where sys._getframe is not defined (Jython for example) or sys._getframe
    # is not defined for arguments greater than 0 (IronPython).
    try:
        result.__module__ = sys._getframe(1).f_globals.get('__name__',
                                                           '__main__')
    except (AttributeError, ValueError):
        pass

    return result
