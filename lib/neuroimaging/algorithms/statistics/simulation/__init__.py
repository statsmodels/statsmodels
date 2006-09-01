import gc

from neuroimaging import traits

from neuroimaging.image import Image, roi

class Simulator(traits.HasTraits):

    intermediate = traits.true
    search = traits.Instance(roi.ROI)
    verbose = traits.false
    parallel = traits.false
    total_iterations = traits.Trait(50, desc='How many iterations?')

    def __init__(self, search=None, verbose=False):

        if search is None:
            self.search = roi.ROIall()
        elif isinstance(search, Image):
            self.search = roi.ROIfromImage(search)
        else:
            self.search = search
        self.verbose = verbose
        try:
            import mpi

            self.parallel = True
        except ImportError:
            self.parallel = False

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

    def simulate(self, n=None, psplit=True, **keywords):
        """
        Run a batch of simulations. If psplit is True and things are running parallel, split the n simulations over
        the processors.
        """

        n = n or self.total_iterations

        values = []
        if self.parallel:
            import mpi

            if psplit:
                nmin = _pmin(n)

                a, b = _prange(n)
                n = b - a
            else:
                nmin = _pmin(n)

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
            if self.parallel:
                if self.intermediate and i < nmin:
                    allvalues = mpi.gather(values)
                    if mpi.rank == 0:
                        self.process(allvalues)
            else:
                if self.intermediate:
                    self.process(values)

        if self.parallel:
            if mpi.rank != 0:
                mpi.finalized()

            allvalues = mpi.gather(values)

            if mpi.rank == 0:
                return self.process(allvalues)
        else:
            return self.process(values)

def _prange(n, rank=None):
    """Parallelize range(n) over mpi.rank processors."""
    t = n / mpi.size

    if t == 0:
        raise ValueError, 'num processors > num in loop -- try fewer processors'
    t = t + 1

    if rank is None:
        rank = mpi.rank

    a = rank * t
    if rank == mpi.size - 1:
        b = n
    else:
        b = a + t
    return a, b

def _pmin(n):
    """
    Determine minimum number of iterations for
    any of the processors.
    """

    a1, b1 = _prange(n, rank=0)
    n1 = b1 - a1

    a2, b2 = _prange(n, rank=mpi.size-1)
    n2 = b2 - a2

    return min(n1, n2)

import rft, resampling

