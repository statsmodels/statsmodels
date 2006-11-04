"""
This module provides various regression analysis techniques to model the
relationship between the dependent and independent variables.
"""

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
        tmp = [data for data in self.iterator]
        for data in tmp:
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
                output.set_next(data=out)


class RegressionOutput(object):

    """
    A generic output for regression. Key feature is that it has
    an \'extract\' method which is called on an instance of
    Results.
    """

    def __init__(self, grid, nout=1):
        self.grid = grid
        self.nout = nout
        self.img = NotImplemented

    def sync_grid(self, img=None):
        """
        Synchronize an image's grid iterator to self.grid's iterator.
        """
        if img is None:
            img = self.img
        img.grid._iterguy = self.grid._iterguy
        iter(img)

    def __iter__(self):
        iter(self.img)
        return self

    def next(self):
        return self.img.next()

    def set_next(self, data):
        self.img.set_next(data)

    def extract(self, results):
        raise NotImplementedError
