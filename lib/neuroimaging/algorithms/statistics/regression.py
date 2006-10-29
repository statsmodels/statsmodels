"""
This module provides various regression analysis techniques to model the
relationship between the dependent and independent variables.
"""

import gc

import numpy as N

class LinearModelIterator(object):

    def __init__(self, iterator, outputs=[]):
        self.iterator = iter(iterator)
        self.outputs = [iter(output) for output in outputs]

    def model(self, **keywords):
        """
        This method should take the iterator at its current state and
        return a LinearModel object.
        """
        return None

    def fit(self, **keywords):
        """
        Go through an iterator, instantiating model and passing data,
        going through outputs.
        """
        tmp = []
        for data in self.iterator:
            tmp.append(data)
        for data in tmp: #self.iterator:
            shape = data.shape[1:]
            data = data.reshape(data.shape[0], N.product(shape))
            model = self.model()

            results = model.fit(data, **keywords)
            for output in self.outputs:
                out = output.extract(results)
                if output.nout > 1:
                    out.shape = (output.nout,) + shape
                else:
                    out.shape = shape
                iter(output)
                output.next(data=out)

            del(results); gc.collect()

class RegressionOutput(object):

    """
    A generic output for regression. Key feature is that it has
    an \'extract\' method which is called on an instance of
    Results.
    """

    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError

    def next(self):
        raise NotImplementedError

    def extract(self, results):
        raise NotImplementedError


