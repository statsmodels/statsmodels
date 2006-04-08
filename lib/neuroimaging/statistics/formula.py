import sets, types, re, string, csv, copy
import numpy as N
import enthought.traits as traits

class Term:

    """
    This class is very simple: it is just a named term in a model formula.
    It is also callable: by default it returns globals()[self.name], but by specifying the argument _fn, this behaviour can be easily changed. For instance, to return the square root of a term, etc.
    The return value is a dictionary with one key (the term name) and value, the output of self.__call__.
    By providing a dictionary for argument 'namespace', one can return something not in the global name space.
    """

    def __init__(self, name, _fn=None, termname=None):
        self.name = name

        if termname is None:
            self.termname = name
        else:
            self.termname = termname

        if type(self.termname) is not types.StringType:
            raise ValueError, 'expecting a string for termname'
        if _fn:
            self._fn = _fn

    def __str__(self):
        return '<term: %s>' % self.termname

    def __add__(self, other):
        """
        Return a Formula object that has the columns self and the columns
        of other.
        """

        other = Formula(other)
        return other + self

    def __mul__(self, other):

        if other.name is 'intercept':
            return Formula(self)
        elif self.name is 'intercept':
            return Formula(other)

        other = Formula(other)
        return other * self

    def names(self):
        if type(self.name) is types.StringType:
            return [self.name]
        else:
            return list(self.name)

    def __call__(self, namespace=None, usefn=True, **extra):
        if namespace is None:
            namespace = globals()
	if not hasattr(self, '_fn') or not usefn:
            val = namespace[self.termname]
            if isinstance(val, Formula):
                val = val.design(namespace)
	else:
            val = self._fn(namespace=namespace, **extra)
        val = N.array(val)
        if len(val.shape) == 1:
            val.shape = (1,) + val.shape
        return N.squeeze(val)

class Factor(Term):

    ordinal = traits.false

    def __init__(self, name, keys, ordinal=False):
        self.keys = list(sets.Set(keys))
        self.keys.sort()
        self._name = name
        self.termname = name
        self.ordinal = ordinal

        if self.ordinal:
            self._sort = True

            def _fn(namespace=None, key=key):
                v = namespace[self._name]
                col = [float(self.keys.index(v[i])) for i in range(n)]
                return N.array(col)
            Term.__init__(self, self.name, _fn=_fn)

        else:
            def _fn(namespace=None):
                v = namespace[self._name]
                value = []
                for key in self.keys:
                    col = [float((v[i] == key)) for i in range(len(v))]
                    value.append(col)
                return N.array(value)
            Term.__init__(self, ['(%s==%s)' % (self.termname, str(key)) for key in self.keys], _fn=_fn, termname=self.termname)

    def __call__(self, namespace=None, values=False, **extra):
        if namespace is None:
            namespace = globals()
        if not values:
            return Term.__call__(self, namespace=namespace, usefn=True, **extra)
        else:
            return Term.__call__(self, namespace=namespace, usefn=False, **extra)

    def _ordinal_changed(self):
        if self.ordinal:
            self.name = self._name
            if not hasattr(self, '_sort'):
                self.keys.sort()
                self._sort = True
        else:
            self.name = [str(key) for key in self.keys]

    def _verify(self, values):
        x = sets.Set(values)
        if not x.issubset(self.keys):
            raise ValueError, 'unknown keys in values'

    def __add__(self, other):
        """
        Return a Formula object that has the columns self and the columns
        of other. When adding \'intercept\' to a Factor, this just returns self.
        """

        if other.name is 'intercept':
            return Formula(self)
        else:
            return Term.__add__(self, other)

    def main_effect(self, reference=None):
        """
        Return the 'main effect' a Term
        that corresponds to the columns in formula.
        """

        if reference is None:
            reference = 0

        def _fn(namespace=None, reference=reference, names=self.names(), **keywords):
            value = N.asarray(self(namespace=namespace, **keywords))
            rvalue = []
            keep = range(value.shape[0])
            keep.pop(reference)
            for i in range(len(keep)):
                rvalue.append(value[keep[i]] - value[reference])
            return rvalue

        keep = range(len(self.names()))
        keep.pop(reference)
        __names = self.names()
        _names = ['%s-%s' % (__names[keep[i]], __names[reference]) for i in range(len(keep))]
        return Term(_names, _fn=_fn, termname='%s:maineffect' % self.termname)

class Quantitative(Term):

    def __pow__(self, power):
        if type(power) is not types.IntType:
            raise ValueError, 'expecting an integer'

        name = '%s^%d' % (self.name, power)

        def _fn(namespace=None, power=power):
            x = N.array(namespace[self.name])
            return N.power(x, power)
        return Term(name, _fn=_fn)

class FuncQuant(Quantitative):

    counter = 0

    def __init__(self, quant, f):
        self.f = f
        self.quant = quant
        def _fn(namespace=None, f=self.f):
            x = namespace[quant.name]
            return f(x)
        try:
            termname = '%s(%s)' % (f.func_name, quant.name)
        except:
            termname = 'f%d(%s)' % (FuncQuant.counter, quant.name)
            FuncQuant.counter += 1
        Term.__init__(self, termname, _fn=_fn)

class Formula(traits.HasTraits):

    """
    This class is meant to emulate something like R's formula object. Formulas can be added, subtracted and multiplied using python's standard order of operations. Essentially it is a list of Terms, as defined above.
    A Formula is callable, again with an optional 'namespace' dictionary argument which a matrix rows whose values are the corresponding values of the terms in the formula.
    """

    terms = traits.List()

    def _terms_changed(self):
        self._names = self.names()
        self._termnames = self.termnames()

    def __init__(self, terms):

        if isinstance(terms, Formula):
            self.terms = copy.copy(list(terms.terms))
        elif type(terms) is types.ListType:
            self.terms = terms
        elif isinstance(terms, Term):
            self.terms = [terms]
        else:
            raise ValueError

        self._terms_changed()

    def __str__(self):
        value = []
        for term in self.terms:
            value += [term.termname]
        return '<formula: %s>' % string.join(value, ' + ')

    def __call__(self, namespace=None, n=-1, **extra):
        if namespace is None:
            namespace = globals()
        allvals = []
        intercept = False
        for term in self.terms:
            val = term(namespace=namespace, **extra)
            if val.shape == ():
                intercept = True
            elif val.ndim == 1:
                val.shape = (1, val.shape[0])
            allvals.append(val)

        if not intercept:
            allvals = N.concatenate(allvals)
        else:
            if allvals != []:
                n = allvals.shape[1]
                allvals = N.concatenate([N.ones((1,n), N.Float), allvals])
            elif n <= 1:
                raise ValueError, 'with no columns in model, keyword n argument needed for intercept'

        return allvals

    def hasterm(self, term):
        """
        Determine whether a given term is in a formula.
        """

        if not isinstance(term, Formula):
            return term.termname in self.termnames()
        elif len(term.terms) == 1:
            term = term.terms[0]
            return term.termname in self.termnames()
        else:
            raise ValueError, 'more than one term passed to hasterm'

    def termcolumns(self, term, dict=False):
        """
        Return a list of the indices of all columns associated
        to a given term.
        """

        if self.hasterm(term):
            names = term.names()
            value = {}
            for name in names:
                value[name] = self._names.index(name)
        else:
            raise ValueError, 'term not in formula'
        if dict:
            return value
        else:
            return value.values()

    def names(self):
        """
        Return a list of the names in the formula. The order of the
        names corresponds to the order of the columns when the formula
        is evaluated.
        """

        allnames = []
        for term in self.terms:
            allnames += term.names()
        return allnames

    def termnames(self):
        """
        Return a list of the term names in the formula. These
        are the names of each \'term\' in the formula.
        """

        names = []
        for term in self.terms:
            names += [term.termname]
        return names

    def design(self, namespace=None, **keywords):
        """
        Given a namespace, return the design matrix (and the column mapping) for a given formula.
        """
        if namespace is None:
            namespace = globals()

        D = N.transpose(self(namespace=namespace, **keywords))

        return D

    def __mul__(self, other, nested=False):
        """
        This returns a formula that is the product of the formula
        of self and that of other.
        Have not implemented a nesting relationship here. Should not be too difficult.
        """

        other = Formula(other)

        selftermnames = self.termnames()
        othertermnames = other.termnames()

        I = len(selftermnames)
        J = len(othertermnames)

        terms = []
        termnames = []

        for i in range(I):
            for j in range(J):
                termname = '%s*%s' % (str(selftermnames[i]), str(othertermnames[j]))
                pieces = termname.split('*')
                pieces.sort()
                termname = string.join(pieces, '*')
                termnames.append(termname)

                selfnames = self.terms[i].names()
                othernames = other.terms[j].names()

                if self.terms[i].name is 'intercept':
                    term = other.terms[j]
                elif other.terms[j].name is 'intercept':
                    term = self.terms[i]

                else:
                    names = []
                    for r in range(len(selfnames)):
                        for s in range(len(othernames)):
                            name = '%s*%s' % (str(selfnames[r]), str(othernames[s]))
                            pieces = name.split('*')
                            pieces.sort()
                            name = string.join(pieces, '*')
                            names.append(name)

                    def _fn(namespace=None, selfterm=self.terms[i], otherterm=other.terms[j], **extra):
                        value = []
                        selfval = N.array(selfterm(namespace=namespace, **extra))
                        if len(selfval.shape) == 1:
                            selfval.shape = (1, selfval.shape[0])
                        otherval = N.array(otherterm(namespace=namespace, **extra))
                        if len(otherval.shape) == 1:
                            otherval.shape = (1, otherval.shape[0])

                        for r in range(selfval.shape[0]):
                            for s in range(otherval.shape[0]):
                                value.append(selfval[r] * otherval[s])

                        return N.array(value)
                    term = Term(names, _fn=_fn, termname=termname)
                terms.append(term)

        return Formula(terms)

    def __add__(self, other):

        """
        Return a Formula with columns of self and columns of other.

        Terms in the formula are sorted alphabetically.
        """

        other = Formula(other)
        terms = self.terms + other.terms
        pieces = [(term.name, term) for term in terms]
        pieces.sort()
        terms = [piece[1] for piece in pieces]
        return Formula(terms)

    def __sub__(self, other):

        """
        Return a Formula with all terms in other removed from self.
        """

        other = Formula(other)
        terms = copy.copy(self.terms)

        for term in other.terms:
            for i in range(len(terms)):
                if terms[i].termname == term.termname:
                    terms.pop(i)
                    break
        return Formula(terms)

def isnested(A, B, namespace=globals()):
    """
    Is factor B nested within factor A or vice versa: a very crude test....

    A and B should are sequences of values here....

    If they are nested, returns (True, F) where F is the finest level of the
    relationship. Otherwise, returns (False, None)

    """

    a = A(namespace, values=True)[0]
    b = B(namespace, values=True)[0]

    if len(a) != len(b):
        raise ValueError, 'A() and B() should be sequences of the same length'

    nA = len(sets.Set(a))
    nB = len(sets.Set(b))
    n = max(nA, nB)

    AB = [(a[i],b[i]) for i in range(len(a))]
    nAB = len(sets.Set(AB))

    if nAB == n:
        if nA > nB:
            F = A
        else:
            F = B
        return (True, F)
    else:
        return (False, None)

I = Term('intercept', _fn=lambda x: N.array(1))
