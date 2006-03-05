import sets, types, re, string, csv, copy
import numpy as N
import enthought.traits as traits

class NamedVariable:

    """
    This class is very simple: it is just a named quantity.
    It is also callable: by default it returns globals()[self.name], but by specifying the argument _fn, this behaviour can be easily changed. For instance, to return the square root of a variable, etc.
    The return value is a dictionary with one key (the variable name) and value, the output of self.__call__.
    By providing a dictionary for argument 'namespace', one can return something not in the global name space.
    """

    def __init__(self, name, _fn=None, varname=None):
        self.name = name

        if varname is None:
            self.varname = name
        else:
            self.varname = varname

        if type(self.varname) is not types.StringType:
            raise ValueError, 'expecting a string for varname'
        if _fn:
            self._fn = _fn

    def __add__(self, other):
        """
        Return a Formula object that has the columns self and the columns
        of other.
        """

        other = Formula(other)
        return other + self

    def __mul__(self, other):

        if other.name is 'intercept':
            return self
        elif self.name is 'intercept':
            return other

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
            val = namespace[self.varname]
            if isinstance(val, Formula):
                val = val.design(namespace)
	else:
            val = self._fn(namespace=namespace, **extra)
        val = N.array(val)
        if len(val.shape) == 1:
            val.shape = (1,) + val.shape
        return val

class Factor(NamedVariable):

    ordinal = traits.false

    def __init__(self, name, keys, ordinal=False):
        self.keys = list(sets.Set(keys))
        self.keys.sort()
        self._name = name
        self.name = name
        self.ordinal = ordinal

        self.variables = {}

        if self.ordinal:
            self._sort = True

            def _fn(namespace, key=key):
                v = namespace[self._name]
                col = [float(self.keys.index(v[i])) for i in range(n)]
                return N.array(col)
            NamedVariable.__init__(self, self.name, _fn=_fn)

        else:
            def _fn(namespace):
                v = namespace[self._name]
                value = []
                for key in self.keys:
                    col = [float((v[i] == key)) for i in range(len(v))]
                    value.append(col)
                return N.array(value)
            NamedVariable.__init__(self, ['(%s==%s)' % (self.name, str(key)) for key in self.keys], _fn=_fn, varname=self.name)

    def __call__(self, namespace=None, values=False, **extra):
        if namespace is None:
            namespace = globals()
        if not values:
            return NamedVariable.__call__(self, namespace=namespace, usefn=True, **extra)
        else:
            return NamedVariable.__call__(self, namespace=namespace, usefn=False, **extra)

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
        of other. When adding "intercept" to a Factor, this just returns self.
        """

        if other.name is 'intercept':
            return self
        else:
            return NamedVariable.__add__(self, other)

class Quantitative(NamedVariable):

    def __pow__(self, power):
        if type(power) is not types.IntType:
            raise ValueError, 'expecting an integer'

        name = '%s**%d' % (self.name, power)

        def _fn(namespace):
            x = N.array(namespace[self.name])
            return pow(x, power)
        return NamedVariable(name, _fn=_fn)

class FuncQuant(Quantitative):

    counter = 0

    def __init__(self, quant, f):
        self.f = f
        self.quant = quant
        def _fn(namespace):
            x = namespace[quant.name]
            return f(x)
        try:
            varname = '%s(%s)' % (f.func_name, quant.name)
        except:
            varname = 'f%d(%s)' % (FuncQuant.counter, quant.name)
            FuncQuant.counter += 1
        NamedVariable.__init__(self, varname, _fn=_fn)

class Formula(traits.HasTraits):

    """
    This class is meant to emulate something like R's formula object. Formulas can be added, subtracted and multiplied using python's standard order of operations. Essentially it is a list of NamedVariables, as defined above.
    A Formula is callable, again with an optional 'namespace' dictionary argument which returns a dictionary with keys given by the variable names, and values the result of evaluating the variables.
    """

    def __init__(self, variables):

        if isinstance(variables, Formula):
            self.variables = copy.copy(variables.variables)
        elif type(variables) is types.ListType:
            self.variables = variables
        elif isinstance(variables, NamedVariable):
            self.variables = [variables]
        else:
            raise ValueError

    def formula(self):
        value = []
        for variable in self.variables:
            value += [variable.varname]
        return string.join(value, ' + ')

    def __call__(self, namespace=None, **extra):
        if namespace is None:
            namespace = globals()
        allvals = []
        intercept = False
        for var in self.variables:
            val = var(namespace=namespace, **extra)
            if len(val.shape) > 1:
                allvals.append(val)
            else:
                intercept = True

        if intercept:
            allvals.append(N.ones((1,n), N.Float))
        return N.concatenate(allvals)

    def hasvariable(self, variable):
        if variable.varname in self.varnames():
            return True
        return False

    def names(self):

        allnames = []
        for var in self.variables:
            allnames += var.names()
        return allnames

    def varnames(self):

        names = []
        for var in self.variables:
            names += [var.varname]
        return names

    def design(self, namespace=globals()):
        """
        Given a namespace, return the design matrix (and the column mapping) for a given formula.
        """
        return N.transpose(self(namespace))

    def __mul__(self, other, nested=False):
        """
        This returns a formula that is the product of the formula
        of self and that of other.
        Have not implemented a nesting relationship here. Should not be too difficult.
        """

        other = Formula(other)

        selfvarnames = self.varnames()
        othervarnames = other.varnames()

        I = len(selfvarnames)
        J = len(othervarnames)

        variables = []
        varnames = []

        for i in range(I):
            for j in range(J):
                varname = '%s*%s' % (str(selfvarnames[i]), str(othervarnames[j]))
                varnames.append(varname)

                selfnames = self.variables[i].names()
                othernames = other.variables[j].names()

                if self.variables[i].name is 'intercept':
                    var = other.variables[j]
                elif other.variables[j].name is 'intercept':
                    var = self.variables[i]

                else:
                    names = []
                    for r in range(len(selfnames)):
                        for s in range(len(othernames)):
                            names.append('%s*%s' % (str(selfnames[r]), str(othernames[s])))
                    def _fn(namespace=None, selfvar=self.variables[i], othervar=other.variables[j], **extra):
                        value = []
                        selfval = N.array(selfvar(namespace=namespace, **extra))
                        otherval = N.array(othervar(namespace=namespace, **extra))

                        for r in range(selfval.shape[0]):
                            for s in range(otherval.shape[0]):
                                value.append(selfval[r] * otherval[s])
                        return N.array(value)
                    var = NamedVariable(names, _fn=_fn, varname=varname)
                variables.append(var)

        return Formula(variables)

    def __add__(self, other):

        """
        Return a Formula with columns of self and columns of other.
        """

        other = Formula(other)
        variables = self.variables + other.variables
        return Formula(variables)

    def __sub__(self, other):

        """
        Return a Formula with all variables in other removed from self.
        """

        other = Formula(other)
        variables = copy.copy(self.variables)

        for var in other.variables:
            for i in range(len(variables)):
                if variables[i].varname == var.varname:
                    variables.pop(i)
                    break
        return Formula(variables)

def clean0(matrix):
    """
    Erase columns of zeros: saves some time in pseudoinverse.
    """
    colsum = add.reduce(matrix**2, 0)

    val = [matrix[:,i] for i in N.nonzero(colsum)[0]]
    return N.array(N.transpose(val))

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

I = NamedVariable('intercept', _fn=lambda x: 1)
