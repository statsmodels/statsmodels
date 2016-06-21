'''Examples for the usage of sandbox/formula

some work, some things don't

'''
from statsmodels.compat.python import iterkeys, zip
import string
import numpy as np

from statsmodels.sandbox import formula
from statsmodels.sandbox import contrast_old as contrast


#Example: Formula from tests, setup

X = np.random.standard_normal((40,10))
namespace = {}
terms = []
for i in range(10):
    name = '%s' % string.ascii_uppercase[i]
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
    raise ValueError('term not in formula')
ValueError: term not in formula


'''
print(form.hasterm('C'))
print(form.termcolumns(formula.Term('C')))  #doesn't work with string argument

#Example: use two columns and get contrast

f2 = (form['A']+form['B'])
print(f2)
print(repr(f2))
list(iterkeys(f2.namespace))   #namespace is still empty
f2.namespace = namespace  #associate data
iterkeys(f2.namespace)
f2.design().shape
contrast.Contrast(formula.Term('A'), f2).matrix

'''
>>> f2 = (form['A']+form['B'])
>>> print f2
<formula: A + B>
>>> print repr(f2)
<statsmodels.sandbox.formula.Formula object at 0x036BAE70>
>>> f2.namespace.keys()   #namespace is still empty
[]
>>> f2.namespace = namespace  #associate data
>>> f2.namespace.keys()
['A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'J']
>>> f2.design().shape
(40, 2)
>>> contrast.Contrast(formula.Term('A'), f2).matrix
array([ 1.,  0.])
'''

#Example: product of terms
#-------------------------

f3 = (form['A']*form['B'])
f3.namespace
f3.namespace = namespace
f3.design().shape
np.min(np.abs(f3.design() - f2.design().prod(1)))

'''
>>> f3 = (form['A']*form['B'])
>>> f3.namespace
{}
>>> f3.namespace = namespace
>>> f3.design().shape
(40,)
>>> np.min(np.abs(f3.design() - f2.design().prod(1)))
0.0
'''

#Example: Interactions of two terms
#----------------------------------

#I don't get contrast of product term

f4 = formula.interactions([form['A'],form['B']])
f4.namespace
f4.namespace = namespace
print(f4)
f4.names()
f4.design().shape

contrast.Contrast(formula.Term('A'), f4).matrix
#contrast.Contrast(formula.Term('A*B'), f4).matrix

'''
>>> formula.interactions([form['A'],form['B']])
<statsmodels.sandbox.formula.Formula object at 0x033E8EB0>
>>> f4 = formula.interactions([form['A'],form['B']])
>>> f4.namespace
{}
>>> f4.namespace = namespace
>>> print f4
<formula: A*B + A + B>
>>> f4.names()
['A*B', 'A', 'B']
>>> f4.design().shape
(40, 3)

>>> contrast.Contrast(formula.Term('A'), f4).matrix
array([  0.00000000e+00,   1.00000000e+00,   7.63278329e-17])
>>> contrast.Contrast(formula.Term('A*B'), f4).matrix
Traceback (most recent call last):
  File "c:\...\scikits\statsmodels\sandbox\contrast_old.py", line 112, in _get_matrix
    self.compute_matrix()
  File "c:\...\scikits\statsmodels\sandbox\contrast_old.py", line 91, in compute_matrix
    T = np.transpose(np.array(t(*args, **kw)))
  File "c:\...\scikits\statsmodels\sandbox\formula.py", line 150, in __call__
    If the term has no 'func' attribute, it returns
KeyError: 'A*B'
'''



#Other
#-----

'''Exception if there is no data or key:
>>> contrast.Contrast(formula.Term('a'), f2).matrix
Traceback (most recent call last):
  File "c:\..\scikits\statsmodels\sandbox\contrast_old.py", line 112, in _get_matrix
    self.compute_matrix()
  File "c:\...\scikits\statsmodels\sandbox\contrast_old.py", line 91, in compute_matrix
    T = np.transpose(np.array(t(*args, **kw)))
  File "c:\...\scikits\statsmodels\sandbox\formula.py", line 150, in __call__
    If the term has no 'func' attribute, it returns
KeyError: 'a'
'''


f = ['a']*3 + ['b']*3 + ['c']*2
fac = formula.Factor('ff', f)
fac.namespace = {'ff':f}


#Example: formula with factor

# I don't manage to combine factors with formulas, e.g. a joint
# designmatrix
# also I don't manage to get contrast matrices with factors
# it looks like I might have to add namespace for dummies myself ?
# even then combining still doesn't work

f5 = formula.Term('A') + fac
namespace['A'] = form.namespace['A']

formula.Formula(fac).design()
'''
>>> formula.Formula(fac).design()
array([[ 1.,  0.,  0.],
       [ 1.,  0.,  0.],
       [ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.],
       [ 0.,  0.,  1.]])


>>> contrast.Contrast(formula.Term('(ff==a)'), fac).matrix
Traceback (most recent call last):
  File "c:\...\scikits\statsmodels\sandbox\contrast_old.py", line 112, in _get_matrix
    self.compute_matrix()
  File "c:\...\scikits\statsmodels\sandbox\contrast_old.py", line 91, in compute_matrix
    T = np.transpose(np.array(t(*args, **kw)))
  File "c:\...\scikits\statsmodels\sandbox\formula.py", line 150, in __call__
    If the term has no 'func' attribute, it returns
KeyError: '(ff==a)'
'''

#convert factor to formula

f7 = formula.Formula(fac)
# explicit updating of namespace with
f7.namespace.update(dict(zip(fac.names(),fac())))

# contrast matrix with 2 of 3 terms
contrast.Contrast(formula.Term('(ff==b)')+formula.Term('(ff==a)'), f7).matrix
#array([[ 1.,  0.,  0.],
#       [ 0.,  1.,  0.]])

# contrast matrix for all terms
contrast.Contrast(f7, f7).matrix
#array([[ 1.,  0.,  0.],
#       [ 0.,  1.,  0.],
#       [ 0.,  0.,  1.]])

# contrast matrix for difference groups 1,2 versus group 0
contrast.Contrast(formula.Term('(ff==b)')+formula.Term('(ff==c)'), f7).matrix - contrast.Contrast(formula.Term('(ff==a)'), f7).matrix
#array([[-1.,  1.,  0.],
#       [-1.,  0.,  1.]])


# all pairwise contrasts
cont = []
for i,j in zip(*np.triu_indices(len(f7.names()),1)):
    ci = contrast.Contrast(formula.Term(f7.names()[i]), f7).matrix
    ci -= contrast.Contrast(formula.Term(f7.names()[j]), f7).matrix
    cont.append(ci)

cont = np.array(cont)
cont
#array([[ 1., -1.,  0.],
#       [ 1.,  0., -1.],
#       [ 0.,  1., -1.]])
