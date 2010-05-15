
import string
import numpy as np

from scikits.statsmodels.sandbox import formula, contrast_old


#Example: Formula from tests

X = np.random.standard_normal((40,10))
namespace = {}
terms = []
for i in range(10):
    name = '%s' % string.uppercase[i]
    namespace[name] = X[:,i]
    terms.append(formula.Term(name))

form = terms[0]
for i in range(1, 10):
    form += terms[i]
form.namespace = namespace

form.design().shape
(40, 10)
'''
>>> dir(form)
['_Formula__namespace', '__add__', '__call__', '__class__',
'__delattr__', '__dict__', '__doc__', '__getattribute__',
'__getitem__', '__hash__', '__init__', '__module__', '__mul__',
'__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__',
'__str__', '__sub__', '__weakref__', '_del_namespace',
'_get_namespace', '_names', '_set_namespace', '_termnames',
'_terms_changed', 'design', 'hasterm', 'names', 'namespace',
'termcolumns', 'termnames', 'terms']

>>> form.design().shape
(40, 10)
>>> form.termnames()
['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
>>> form.namespace.keys()
['A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'J']
>>> form.names()
['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

>>> form.termcolumns(formula.Term('C'))
[2]
>>> form.termcolumns('C')
Traceback (most recent call last):
  File "<pyshell#21>", line 1, in <module>
    form.termcolumns('C')
  File "C:\Josef\eclipsegworkspace\statsmodels-josef-experimental\scikits\statsmodels\sandbox\formula.py", line 494, in termcolumns
    raise ValueError, 'term not in formula'
ValueError: term not in formula


'''




