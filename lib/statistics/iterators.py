import gc
import enthought.traits as traits
from numarray import product

class LinearModelIterator(traits.HasTraits):

    iterator = traits.Any()
    outputs = traits.Any()

    def __init__(self, iterator, outputs=[], **keywords):
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

        for data in self.iterator:
            shape = data.shape[1:]
            data.shape = (data.shape[0], product(shape))
            model = self.model()

            results = model.fit(data, **keywords)
            for output in self.outputs:
                out = output.extract(results)
                if output.nout > 1:
                    out.shape = (output.nout,) + shape
                else:
                    out.shape = shape
                output.next(data=out)

            del(results); gc.collect()
            del(data); gc.collect()
