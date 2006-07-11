import gc

from neuroimaging import traits

from neuroimaging.image import Image, roi

class Simulator(traits.HasTraits):

    intermediate = traits.true
    search = traits.Instance(roi.ROI)
    verbose = traits.false

    def __init__(self, search=None, verbose=False):

        if search is None:
            self.search = roi.ROIall()
        elif isinstance(search, Image):
            self.search = roi.ROIfromImage(search)
        else:
            self.search = search
        self.verbose = verbose

    def generate(self, **keywords):
        """
        Default generate method: returns None.
        This class should be subclassed and this method written over.
        """
        return

    def process(self, values, **keywords):
        """
        Default processing: return simulated values.
        This class should be subclassed and this method written over.
        """
        return values

    def feature(self, image, **keywords):
        """
        Default feature: return None.
        This class should be subclassed and this method written over.
        """
        return

    def simulate(self, n=50, psplit=True, **keywords):
        """
        Run a batch of simulations. If psplit is True and things are running parallel, split the n simulations over
        the processors.
        """

        values = []
        if Options.parallel:
            import mpi

            if psplit:
                nmin = pmin_n(n)

                a, b = prange(n)
                n = b - a

            elif hasattr(self, 'itotal'):
                nmin = pmin_n(self.itotal)
            else:
                nmin = pmin_n(n)
        #
        # slightly silly way to synchronize output,
        # by resending values each time. if values are large
        # this may take a while. BUT, it allows you
        # easy access to partial results, if, for instance
        # you are plotting them as they come in ....
        #
        # for most features, it seems that values
        # will be small.
        #

        for i in range(n):
            field = self.generate(**keywords)
            values.append(self.feature(field, **keywords))
            if Options.parallel:
                if self.intermediate and i < nmin:
                    allvalues = mpi.gather(values)
                    if mpi.rank == 0:
                        self.process(allvalues)
            else:
                if self.intermediate:
                    self.process(values)

        if Options.parallel:
            if mpi.rank != 0:
                mpi.finalized()

            allvalues = mpi.gather(values)

            if mpi.rank == 0:
                return self.process(allvalues)
        else:
            return self.process(values)


from RFT import *
from Resampling import *
