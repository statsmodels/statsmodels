import enthought.traits as traits

class Model(traits.HasTraits):
    def initialize(self, **keywords): pass
    def fit(self, **keywords): pass
    def predict(self, **keywords):
        # results of fit are stored in self.results
        self.results.predict(**keywords)
    def view(self, **keywords): pass

import unittest
def suite():
    return unittest.TestSuite([tests.suite()])

