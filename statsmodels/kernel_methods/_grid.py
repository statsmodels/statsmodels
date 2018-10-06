from __future__ import division, print_function, absolute_import
import numpy as np
from numpy.lib.stride_tricks import broadcast_arrays
from ..compat.python import range, zip


class Grid(object):
    """
    Object representing a grid.

    Can be converted to a full array using "np.array" function.

    Parameters
    ----------
    grid_axes: list of ndarray
        Each ndarray can have at most 1 dimension with more than 1 element.
        This dimension contains the position of each point on the axis
    bounds: ndarray
        This is a Dx2 array. For each dimension, the lower and upper bound of
        the axes (doesn't have to correspond to the min and max of the axes.
    bin_type: str
        A string with as many letter as there are dimensions. For each
        dimension, gives the kind of axes. Can be one of 'b', 'r', 'c' or 'd'
        (See :py:attr:`bin_type`). If a single letter is provided, this is the
        class for all axes. If not specified, the default is 'b'.
    edges: list of ndarray
        If provided, should be a list with one array per dimension. Each array
        should have one more element than the bin for that dimension. These
        represent the edges of the bins.
    """
    def __init__(self, grid_axes, bounds=None, bin_type=None, edges=None, dtype=None):
        self._interval = None
        if isinstance(grid_axes, Grid):
            if bounds is None:
                bounds = grid_axes.bounds
            if bin_type is None:
                bin_type = grid_axes.bin_type
            if edges is None and grid_axes._edges is not None:
                edges = grid_axes._edges
            self.__init__(grid_axes.grid, bounds, bin_type, edges, dtype)
            return
        first_elemt = np.asarray(grid_axes[0])
        if first_elemt.ndim == 0:
            ndim = 1
            grid_axes = [np.asarray(grid_axes)]
        else:
            ndim = len(grid_axes)
            grid_axes = [np.asarray(ax) for ax in grid_axes]
        if dtype is None:
            dtype = np.find_common_type([ax.dtype for ax in grid_axes], [])
        for d in range(ndim):
            if grid_axes[d].ndim != 1:
                raise ValueError("Error, the axis of a grid must be 1D arrays or "
                                 "have exacltly one dimension with more than 1 element")
            grid_axes[d] = grid_axes[d].astype(dtype)
        self._grid = grid_axes
        self._ndim = ndim
        if bin_type is None:
            bin_type = 'b' * ndim
        else:
            bin_type = str(bin_type)
        if len(bin_type) == 1:
            bin_type = bin_type * ndim
        elif len(bin_type) != ndim:
            raise ValueError("Error, there must be as many bin_type as dimensions")
        if any(b not in 'crbd' for b in bin_type):
            raise ValueError("Error, bin type must be one of 'b', 'r', 'c' or 'd'")
        self._bin_type = bin_type
        self._shape = tuple(len(ax) for ax in grid_axes)
        if edges is not None:
            edges = [e.astype(dtype) for e in edges]
        self._edges = edges

        expected_bounds = np.empty((ndim, 2), dtype=dtype)
        for d in range(ndim):
            ax = grid_axes[d]
            if bin_type[d] == 'd':
                expected_bounds[d] = [ax[0], ax[-1]]
            else:
                expected_bounds[d] = [(3 * ax[0] - ax[1]) / 2, (3 * ax[-1] - ax[-2]) / 2]

        if bounds is None:
            bounds = expected_bounds
        else:
            bounds = np.asarray(bounds)
            if bounds.ndim == 1:
                bounds = bounds[None, :]
            if (bounds[:, 0] >= bounds[:, 1]).any():
                raise ValueError("The lower bounds must be strictly smaller than the upper bounds")
            if bounds.shape != expected_bounds.shape:
                raise ValueError("Bounds must be a (D,2) array with D the dimension of the grid")
        self._bounds = bounds

    def __repr__(self):
        dims = 'x'.join(str(s) + bt for s, bt in zip(self.shape, self.bin_type))
        lims = '[{}]'.format(" ; ".join('{0:g} - {1:g}'.format(b[0], b[1]) for b in self.bounds))
        return "<Grid {0}, {1}, dtype={2}>".format(dims, lims, self.dtype)

    def copy(self):
        """
        Deep-copy the content of the grid
        """
        bounds = self._bounds.copy()
        return Grid(self, bounds=bounds)

    @staticmethod
    def fromSparse(grid, *args, **kwords):
        """
        Create a grid from a sparse mesh.

        Parameters
        ----------
        grid: list of ndarray
            This is the result of using `ogrid` or `meshgrid` with `sparse` set to `True`.

        Other arguments are passed to the constructor.
        """
        return Grid([np.squeeze(g) for g in grid], *args, **kwords)

    @staticmethod
    def fromFull(grid, order='F', *args, **kwords):
        """
        Create a Grid from a full mesh represented as a single ndarray.
        """
        grid_shape = None
        if order == 'F':
            grid_shape = grid.shape[:-1]
            ndim = grid.shape[-1]
        else:
            grid_shape = grid.shape[1:]
            ndim = grid.shape[0]
        if len(grid_shape) != ndim:
            raise ValueError("This is not a valid grid")
        grid_axes = [None] * ndim
        selector = [0] * ndim
        for d in range(ndim):
            selector[d] = np.s_[:]
            if order == 'F':
                sel = tuple(selector) + (d,)
            else:
                sel = (d,) + tuple(selector)
            grid_axes[d] = grid[sel]
            selector[d] = 0
        return Grid(grid_axes, *args, **kwords)

    @staticmethod
    def fromArrays(grid, *args, **kwords):
        """
        Create a grid from a list of grids, a list of arrays or a full array C or Fortram-style.
        """
        try:
            grid = np.asarray(grid).squeeze()
            if not np.issubdtype(grid.dtype, np.number):
                raise ValueError('Argument is not a full numeric grid')
            if grid.ndim == 2:  # Cannot happen for full grid
                raise ValueError('Argument is not a full numeric grid')
            if grid.ndim == 1:
                return Grid.fromFull(grid, 'C', *args, **kwords)
            ndim = grid.ndim - 1
            if grid.shape[-1] == ndim:
                return Grid.fromFull(grid, 'F', *args, **kwords)
            elif grid.shape[0] == ndim:
                return Grid.fromFull(grid, 'C', *args, **kwords)
        except ValueError:
            return Grid.fromSparse(grid, *args, **kwords)
        raise ValueError("Couldn't find what kind of grid this is.")

    @staticmethod
    def fromBounds(bounds, bin_type='b', shape=256, **kwargs):
        """
        Create a grid from bounds, types and bin sizes.

        Parameters
        ----------
        bounds: array-like of shape (D,2)
            For each axis, the lower and upper bound
        bin_type: str
            String defining the bin types. If a single character is used, all
            axis will have this type. As usual, the types may be one of 'c',
            'r', 'b' or 'd'
        shape: int or list of int
            If a single int, all axis will have this size, otherwise, specify a
            size per axis. Note that discrete axis are always the range from
            low to high, so their bin size is ignored.
        kwargs: dict
            Any extra argument is forwarded to the Grid constructor

        Returns
        -------
        A Grid object with the given characteristics
        """
        bounds = np.atleast_2d(bounds)
        if bounds.shape[1] != 2:
            raise ValueError("Bounds must be a (D,2) array for a D-dimensional grid")
        ndim = bounds.shape[0]
        if len(bin_type) == 1:
            bin_type = bin_type * ndim
        if any(b not in 'crbd' for b in bin_type):
            raise ValueError("A bin type must be one of 'c', 'r' , 'b' and 'd'")
        shape = np.asarray(shape, dtype=int)
        if not shape.shape:
            shape = shape * np.ones((ndim,), dtype=int)
        elif shape.shape != (ndim,):
            raise ValueError("Shape must be either a single integer, or an integer per dimension")
        grid = [None]*ndim
        for d in range(ndim):
            if bin_type[d] == 'd':
                grid[d] = np.arange(bounds[d, 0], bounds[d, 1]+1)
            else:
                dx = (bounds[d, 1] - bounds[d, 0]) / shape[d]
                grid[d] = np.linspace(bounds[d, 0] + dx/2,
                                      bounds[d, 1] - dx/2,
                                      shape[d])
        return Grid(grid, bounds, bin_type, **kwargs)

    @property
    def ndim(self):
        """
        Number of dimensions of the grid
        """
        return self._ndim

    @property
    def bin_type(self):
        """
        Types of the axes.

        The valid types are:
            - b: Bounded -- The axes is on a bounded domain. During binning,
                any values outside the domain is ignored, while during
                interpolation, such values are put back to the closest
                boundary.
            - r: Reflective -- The axes are infinite, but the data are
                'reflected' at the boundaries. Any point outside the domain is
                put back inside by reflection on the boundaries.
            - c: Cyclic -- The axes are infinite, but the data is cyclic. Any
                point outside the defined domain is put back inside using the
                cyclic nature of the function.
            - d: Discrete -- The axes are finite and non-continuous. Any data
                outside the domain is ignored both for interpolation and
                binning. Also, interpolation is to the nearest value.

        Notes
        -----
        If bin_type is specified with a single letter, all dimensions will be given the new type.
        """
        return self._bin_type

    @bin_type.setter
    def bin_type(self, bin_type):
        bin_type = str(bin_type)
        if any(c not in 'brcd' for c in bin_type):
            raise ValueError("Error, the letters in 'bin_type' must be one of 'brcd'")
        if len(bin_type) == 1:
            bin_type = bin_type * self.ndim
        if len(bin_type) != self.ndim:
            raise ValueError("Error, 'bin_type' must have either one letter or one letter per dimension")
        self._bin_type = bin_type

    @property
    def shape(self):
        """
        Shape of the grid (e.g. number of bin for each dimension)
        """
        return self._shape

    @property
    def edges(self):
        """
        list of ndarray
            Edges of the bins for each dimension
        """
        if self._edges is None:
            edges = [np.empty((s + 1,), dtype=self.dtype) for s in self._shape]
            for d, (es, bnd, ax, bn) in enumerate(zip(edges, self.bounds, self.grid, self.bin_type)):
                if bn == 'd':
                    es[:] = np.arange(len(ax) + 1) - 0.5
                else:
                    es[1:-1] = (ax[1:] + ax[:-1]) / 2
                    es[0] = bnd[0]
                    es[-1] = bnd[1]
            self._edges = edges
        return self._edges

    @property
    def grid(self):
        """
        list of ndarray
            Position of the bins for each dimensions
        """
        return tuple(self._grid)

    @property
    def dtype(self):
        """
        Type of arrays for the bin positions
        """
        return self._grid[0].dtype

    @property
    def bounds(self):
        """
        ndarray
            Dx2 array with the bounds of each axes
        """
        return self._bounds

    @property
    def start_interval(self):
        """
        For each dimension, the distance between the two first edges, or if
        there are no edges, the distance between the two first bins (which will
        be the same thing ...).
        """
        if self._interval is None:
            ndim = self.ndim
            if self._edges is not None:
                axes = self._edges
            else:
                axes = self.grid
            inter = np.empty((ndim,), dtype=self.dtype)
            for d in range(ndim):
                inter[d] = axes[d][1] - axes[d][0]
            self._interval = inter
        return self._interval

    def bin_sizes(self):
        """
        Return the size of each bin, per dimension.

        Notes: this requires computed edges if they are not already present
        """
        edges = self.edges
        return [es[1:] - es[:-1] for es in edges]

    @property
    def start_volume(self):
        """
        Return the volume of the first bin, using :py:attr:`start_interval`
        """
        return np.prod(self.start_interval)

    def bin_volumes(self):
        """
        Return the volume of each bin
        """
        if self.ndim == 1:
            return self.bin_sizes()[0]
        bins = np.meshgrid(*self.bin_sizes(), indexing='ij', copy=False, sparse=True)
        return np.prod(bins)

    def full(self, order='F'):
        """
        Return a full representation of the grid.

        If order is 'C', then the first index is the dimension, otherwise the last index is.
        """
        if self._ndim == 1:
            return self._grid[0]
        m = broadcast_arrays(*np.meshgrid(*self._grid, indexing='ij', sparse='True', copy='False'))
        if order is 'C':
            return np.asarray(m)
        return np.concatenate([mm[..., None] for mm in m], axis=-1)

    def __array__(self):
        """
        Convert as a full array
        """
        return self.full()

    def linear(self):
        """
        Return a 2D array with all the points "in line"
        """
        if self._ndim == 1:
            return self._grid[0]
        m = self.full()
        npts = np.prod(self.shape)
        return m.reshape(npts, self.ndim)

    def sparse(self):
        """
        Return the sparse representation of the grid.
        """
        if self._ndim == 1:
            return [self._grid[0]]
        return np.meshgrid(*self._grid, indexing='ij', copy=False, sparse=True)

    def __iter__(self):
        return iter(self._grid)

    def __len__(self):
        return len(self._grid)

    def __getitem__(self, idx):
        """
        Shortcut to access bin positions.

        Usage
        -----
        >>> grid = Grid([[1,2,3,4,5],[-1,0,1],[7,8,9,10]])
        >>> grid[0,2]
        3
        >>> grid[2,:2]
        array([7,8])
        >>> grid[1]
        array([-1,0,1])
        """
        try:
            dim, pos = idx
            return self._grid[dim][pos]
        except TypeError:
            return self._grid[idx]

    def transform(self, fcts):
        '''
        Transform an axis of the grid, in place.

        Parameters
        ----------
        fcts: fun or list of fun or dict of int: fun
            Either a single function, a list with one function per dimension or
            a dictionnary giving, for a set of axis, how to transform them.
        '''
        if callable(fcts):
            fcts = [fcts] * self.ndim
        for i in range(self.ndim):
            try:
                f = fcts[i]
            except (IndexError, KeyError):
                pass
            else:
                if f is not None:
                    self.edges[i][...] = f(self.edges[i])
                    self.bounds[i][...] = f(self.bounds[i])
                    self.grid[i][...] = f(self.grid[i])
        return self

    def integrate(self, values=None):
        """
        Integrate values over the grid

        If values is None, the integration is of the function f(x) = 1
        """
        if values is None:
            return np.sum(self.bin_volumes())
        values = np.asarray(values)
        return np.sum(values * self.bin_volumes())

    def cum_integrate(self, values=None):
        """
        Integrate values over the grid and return the cumulative values

        If values is None, the integration is of the function f(x) = 1
        """
        if values is None:
            out = self.bin_volumes()
        else:
            values = np.asarray(values)
            out = values * self.bin_volumes()
        for d in range(self.ndim):
            out.cumsum(axis=d, out=out)
        return out

    def __eq__(self, other):
        if not isinstance(other, Grid):
            return NotImplemented
        if self.shape != other.shape:
            return False
        if self.bin_type != other.bin_type:
            return False
        if (self.bounds != other.bounds).any():
            return False
        if any((g1 != g2).any() for (g1, g2) in zip(self.grid, other.grid)):
            return False
        if self._edges is not None or other._edges is not None:
            if any((g1 != g2).any() for (g1, g2) in zip(self.edges, other.edges)):
                return False
        return True

    def almost_equal(self, other, rtol=1e-6, atol=1e-8):
        """
        Check for two grids to be almost equal

        Parameters
        ----------
        other: Grid
            Grid to compare
        rtol: float
            Relative tolerance
        atol: float
            Absolute tolerance. If None, this will be the same as rtol
        """
        if atol is None:
            atol = rtol
        if self.shape != other.shape:
            return False
        if self.bin_type != other.bin_type:
            return False
        if not np.allclose(self.bounds, other.bounds, rtol, atol):
            return False
        if any(not np.allclose(g1, g2, rtol, atol) for (g1, g2) in zip(self.grid, other.grid)):
            return False
        if self._edges is not None or other._edges is not None:
            if any(not np.allclose(g1, g2, rtol, atol) for (g1, g2) in zip(self.edges, other.edges)):
                return False
        return True
