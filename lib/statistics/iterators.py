import gc
from LinearModel import LinearModel
import enthought.traits as TR
from numarray import product

class LinearModelIterator(TR.HasTraits):

    def __init__(self, iterator, design, outputs=[], **keywords):
        self.iterator = iter(iterator)
        self.design = design
        self.outputs = [iter(output) for output in outputs]

    def model(self, **keywords):
        """
        This method should take the iterator at its current state and
        return a LinearModel object.
        """
        return self.design.model(**keywords)

    def fit(self, **keywords):
        """
        Go through an iterator, instantiating model and passing data,
        going through outputs.
        """

        for data in self.iterator:
            shape = data.shape[1:]
            data.setshape((data.shape[0], product(shape)))
            model = self.model()

            results = model.fit(data, **keywords)
            for output in self.outputs:
                out = output.extract(results)
                if output.ndim > 1:
                    out.setshape((output.ndim,) + shape)
                else:
                    out.setshape(shape)
                output.next(data=out, iterator=self.iterator)

            del(results); gc.collect()
            del(data); gc.collect()
