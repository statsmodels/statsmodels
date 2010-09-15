'''

Formulas
--------

This follows mostly Greene notation (in slides)
partially ignoring factors tau or mu for now, ADDED
(if all tau==1, then runmnl==clogit)

leaf k probability :

Prob(k|j) = exp(b_k * X_k / mu_j)/ sum_{i in L(j)} (exp(b_i * X_i / mu_j)

branch j probabilities :

Prob(j) = exp(b_j * X_j + mu*IV_j )/ sum_{i in NB(j)} (exp(b_i * X_i + mu_i*IV_i)

inclusive value of branch j :

IV_j = log( sum_{i in L(j)} (exp(b_i * X_i / mu_j) )

this is the log of the denominator of the leaf probabilities


L(j) : leaves at branch j, where k is child of j
NB(j) : set of j and it's siblings

Design
------

* splitting calculation transmission between returns and changes to
  instance.probs
  - probability for each leaf is in instance.probs
  - inclusive values and contribution of exog on branch level need to be
    added separately. handed up the tree through returns


bugs/problems
-------------

* singleton branches return zero to `top`, not a value
  I'm not sure what they are supposed to return, given the split between returns
  and instance.probs
* Why does 'Air' (singleton branch) get probability exactly 0.5 ?


'''

import numpy as np
from pprint import pprint


testxb = 2 #global to class to return strings instead of numbers

class RU2NMNL(object):
    '''Nested Multinomial Logit with Random Utility 2 parameterization

    '''

    def __init__(self, endog, exog, tree, paramsind):
        self.endog = endog
        self.datadict = exog
        self.tree = tree
        self.paramsind = paramsind

        self.branchsum = ''
        self.probs = {}
        self.probstxt = {}
        self.branchleaves = {}
        self.branchvalues = {}  #just to keep track of returns by branches

        #copied over but not quite sure yet
        #unique, parameter array names,
        #sorted alphabetically, order is/should be only internal
        self.paramsnames = sorted(set([i for j in paramsind.values() for i in j]))

        #mapping coefficient names to indices to unique/parameter array
        self.paramsidx = dict((name, idx) for (idx,name) in
                              enumerate(self.paramsnames))

        #mapping branch and leaf names to index in parameter array
        self.parinddict = dict((k, [self.paramsidx[j] for j in v])
                               for k,v in self.paramsind.items())

        self.recursionparams = 1. + np.arange(len(self.paramsnames))
        #for testing that individual parameters are used in the right place
        self.recursionparams = np.zeros(len(self.paramsnames))
        self.recursionparams[1] = 1




    def calc_prob(self, tree, parent=None):
        '''walking a tree bottom-up based on dictionary
        '''
        endog = self.endog
        datadict = self.datadict
        paramsind = self.paramsind
        branchsum = self.branchsum


        if type(tree) == tuple:   #assumes leaves are int for choice index
            name, subtree = tree
            self.branchleaves[name] = []  #register branch in dictionary
            print name, datadict[name]
            print 'subtree', subtree
            branchvalue = []
            if testxb:
                branchsum = datadict[name]
            else:
                branchsum = name
            for b in subtree:
                print b
                bv = self.calc_prob(b, name)
                branchvalue.append(bv)
                branchsum = branchsum + bv
            self.branchvalues[name] = branchvalue #keep track what was returned
            print 'branchsum', branchsum

            if parent:
                print 'parent', parent
                self.branchleaves[parent].extend(self.branchleaves[name])
            if 1:  #not name == 'top':
                tmpsum = 0
                for k in self.branchleaves[name]:
                    #similar to this is now also in return branch values
                    #depends on what will be returned
                    tmpsum += self.probs[k]
                    iv = np.log(tmpsum)

                for k in self.branchleaves[name]:
                    self.probstxt[k] = self.probstxt[k] + ['*' + name + '-prob' +
                                    '(%s)' % ', '.join(self.paramsind[name])]

                    self.probs[k] = self.probs[k] / tmpsum
                    if np.size(self.datadict[name])>0:
                        #self.probs[k] = self.probs[k] / tmpsum
##                            np.exp(-self.datadict[name] *
##                             np.sum(self.recursionparams[self.parinddict[name]]))
                        print 'self.datadict[name], self.probs[k]',
                        print self.datadict[name], self.probs[k]
                    #if not name == 'top':
                    #    self.probs[k] = self.probs[k] * np.exp( iv)

            print 'working on branch', tree, branchsum
            if testxb<2:
                return branchsum
            else:
                return iv

        else:
            print 'parent', parent
            self.branchleaves[parent].append(tree) # register leave with parent
            self.probstxt[tree] = [tree + '-prob' +
                                '(%s)' % ', '.join(self.paramsind[tree])]
            self.probs[tree] = np.exp(np.sum(self.datadict[tree] *
                                  self.recursionparams[self.parinddict[tree]]))

            if testxb == 2:
                return self.probs[tree]
            elif testxb == 1:
                leavessum = np.array(datadict[tree]) # sum((datadict[bi] for bi in datadict[tree]))
                print 'final branch with', tree, ''.join(tree), leavessum #sum(tree)
                return leavessum  #sum(xb[tree])
            elif testxb == 0:
                return ''.join(tree) #sum(tree)





endog = 5 # dummy place holder


##############  Example similar to Greene

#get pickled data
#endog3, xifloat3 = pickle.load(open('xifloat2.pickle','rb'))


tree0 = ('top',
            [('Fly',['Air']),
             ('Ground', ['Train', 'Car', 'Bus'])
             ]
        )

''' this is with real data from Greene's clogit example
datadict = dict(zip(['Air', 'Train', 'Bus', 'Car'],
                    [xifloat[i]for i in range(4)]))
'''

#for testing only (mock that returns it's own name
datadict = dict(zip(['Air', 'Train', 'Bus', 'Car'],
                    ['Airdata', 'Traindata', 'Busdata', 'Cardata']))

if testxb:
    datadict = dict(zip(['Air', 'Train', 'Bus', 'Car'],
                    np.arange(4)))

datadict.update({'top' :   [],
                 'Fly' :   [],
                 'Ground': []})

paramsind = {'top' :   [],
             'Fly' :   [],
             'Ground': [],
             'Air' :   ['GC', 'Ttme', 'ConstA', 'Hinc'],
             'Train' : ['GC', 'Ttme', 'ConstT'],
             'Bus' :   ['GC', 'Ttme', 'ConstB'],
             'Car' :   ['GC', 'Ttme']
             }

modru = RU2NMNL(endog, datadict, tree0, paramsind)

print 'Example 1'
print '---------\n'
print modru.calc_prob(modru.tree)

print 'Tree'
pprint(modru.tree)
print '\nmodru.probs'
pprint(modru.probs)



##############  example with many layers

tree2 = ('top',
            [('B1',['a','b']),
             ('B2',
                   [('B21',['c', 'd']),
                    ('B22',['e', 'f', 'g'])
                    ]
              ),
             ('B3',['h'])
            ]
         )

#Note: dict looses ordering
paramsind2 = {
 'B1': [],
 'a': ['consta', 'p'],
 'b': ['constb', 'p'],
 'B2': ['const2', 'x2'],
 'B21': [],
 'c': ['constc', 'p', 'time'],
 'd': ['constd', 'p', 'time'],
 'B22': ['x22'],
 'e': ['conste', 'p', 'hince'],
 'f': ['constf', 'p', 'hincf'],
 'g': [          'p', 'hincg'],
 'B3': [],
 'h': ['consth', 'p', 'h'],
 'top': []}

datadict2 = dict([i for i in zip('abcdefgh',range(8))])
datadict2.update({'top':1000, 'B1':100, 'B2':200, 'B21':21,'B22':22, 'B3':300})
'''
>>> pprint(datadict2)
{'B1': 100,
 'B2': 200,
 'B21': 21,
 'B22': 22,
 'B3': 300,
 'a': 0,
 'b': 1,
 'c': 2,
 'd': 3,
 'e': 4,
 'f': 5,
 'g': 6,
 'h': 7,
 'top': 1000}
'''


modru2 = RU2NMNL(endog, datadict2, tree2, paramsind2)
print '\n\nExample 2'
print '---------\n'
print modru2.calc_prob(modru2.tree)
print 'Tree'
pprint(modru2.tree)
print '\nmodru.probs'
pprint(modru2.probs)

print 'sum of probs', sum(modru2.probs.values())
print 'branchvalues'
print modru2.branchvalues
print modru.branchvalues
