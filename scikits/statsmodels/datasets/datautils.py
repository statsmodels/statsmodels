import os
import time
import numpy as np
from numpy import genfromtxt, array

class Dataset(dict):
    def __init__(self, **kw):
        dict.__init__(self,kw)
        self.__dict__ = self
# Some datasets have string variables. If you want a raw_data attribute you
# must create this in the dataset's load function.
        try: # some datasets have string variables
            self.raw_data = self.data.view((float, len(self.names)))
        except:
            pass

    def __repr__(self):
        return str(self.__class__)
