

from pprint import pprint


testxb = 0 #global to class to return strings instead of numbers

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
        self.branchleaves = {}


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
            keys = []
            if testxb:
                branchsum = datadict[name]
            else:
                branchsum = name
            for b in subtree:
                print b
                branchsum = branchsum + self.calc_prob(b, name)
            print 'branchsum', branchsum, keys

            if parent:
                print 'parent', parent
                self.branchleaves[parent].extend(self.branchleaves[name])
            for k in self.branchleaves[name]:
                self.probs[k] = self.probs[k] + ['*' + name + '-prob']

        else:
            print 'parent', parent
            self.branchleaves[parent].append(tree) # register leave with parent
            self.probs[tree] = [tree + '-prob' +
                                '(%s)' % ', '.join(self.paramsind[tree])]
            if testxb:
                leavessum = sum((datadict[bi] for bi in tree))
                print 'final branch with', tree, ''.join(tree), leavessum #sum(tree)
                return leavessum  #sum(xb[tree])
            else:
                return ''.join(tree) #sum(tree)

        print 'working on branch', tree, branchsum
        return branchsum



endog = 5 # dummy place holder


##############  Example similar to Greene


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
