from BrainSTAT import VImage
from BrainSTAT.Simulation import Simulator
import enthought.traits as TR
import scipy

class Bootstrap(Simulator, TR.HasTraits):
    """
    A class to do bootstrap simulation, given a model and a sequence of images.
    This class should be subclassed, writing over the generate class.

    """

    def __init__(self, sample, search=None, verbose=False, **keywords):

        Simulator.__init__(self, search=search, verbose=verbose)
        if isinstance(sample, VImage):
            self.sample = [sample.toarray(slice=(i,)) for i in range(sample.shape[0])]
        else:
            self.sample = sample

        self.keywords = keywords
        self.nimages = len(self.sample)
        self.original = self.feature(self.model(range(self.nimages)), **keywords)

    def generate(self, **keywords):
        """
        Generate a bootstrap sample.
        """

        _sample = scipy.stats.random_integers(self.nimages-1, min=0, size=self.nimages)
        return self.model(_sample, **self.keywords)

    def process(self, values, **keywords):
        """
        Return original result and bootstrap samples.
        """
        return self.original, values

    def feature(self, value, **keywords):
        """
        Default feature: return None.
        This class should be subclassed and this method written over.
        """
        return

    def model(self, sample, **keywords):
        """
        Default sample: return None.
        This class should be subclassed and this method written over.
        """
        return


