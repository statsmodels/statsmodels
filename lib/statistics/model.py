import numpy as N
import numpy.linalg as NL
import enthought.traits as traits

class Model:

    def __init__(self):
        pass

    def initialize(self, **keywords):
        pass

    def fit(self, **keywords):
        pass

    def predict(self, **keywords):
        self.results.predict(**keywords) # results of fit are stored in self.results

    def view(self, **keywords):
        pass

from regression import OLSModel
from classification import Classifier
