"""Create a mosaic plot from a contingency table.

It allows to visualize multivariate categorical data in a rigorous
and informative way.

for more information you can read:
    http://www.math.yorku.ca/SCS/Online/mosaics/about.html
    http://www.theusrus.de/blog/understanding-mosaic-plots/
    http://www.vicc.org/biostatistics/LuncheonTalks/DrTsai2.pdf
"""
# Author: Enrico Giampieri - 21 Jan 2013

from __future__ import division

import numpy
from collections import OrderedDict
from itertools import product

from numpy import iterable, r_, cumsum, array
from statsmodels.graphics import utils

__all__ = ["mosaic", "hierarchical_split"]


def _normalize_split(proportion):
    """
    return a list of proportions of the available space given the division
    if only a number is given, it will assume a split in two pieces
    """
    if not iterable(proportion):
        if proportion == 0:
            proportion = array([0.0, 1.0])
        elif proportion >= 1:
            proportion = array([1.0, 0.0])
        elif proportion < 0:
            raise ValueError("proportions should be positive,"
                              "given value: {}".format(proportion))
        else:
            proportion = array([proportion, 1.0 - proportion])
    proportion = numpy.asarray(proportion, dtype=float)
    if numpy.any(proportion < 0):
        raise ValueError("proportions should be positive,"
                          "given value: {}".format(proportion))
    if numpy.allclose(proportion, 0):
        raise ValueError("at least one proportion should be"
                          "greater than zero".format(proportion))
    # ok, data are meaningful, so go on
    if len(proportion) < 2:
        return array([0.0, 1.0])
    left = r_[0, cumsum(proportion)]
    left /= left[-1] * 1.0
    return left


def _split_rect(x, y, width, height, proportion, horizontal=True, gap=0.05):
    """
    Split the given rectangle in n segments whose proportion is specified
    along the given axis if a gap is inserted, they will be separated by a
    certain amount of space, retaining the relative proportion between them
    a gap of 1 correspond to a plot that is half void and the remaining half
    space is proportionally divided among the pieces.
    """
    x, y, w, h = float(x), float(y), float(width), float(height)
    if (w < 0) or (h < 0):
        raise ValueError("dimension of the square less than"
                          "zero w={} h=()".format(w, h))
    proportions = _normalize_split(proportion)
    # extract the starting point and the dimension of each subdivision
    # in respect to the unit square
    starting = proportions[:-1]
    amplitude = proportions[1:] - starting
    # how much each extrema is going to be displaced due to gaps
    starting += gap * numpy.arange(len(proportions) - 1)
    # how much the squares plus the gaps are extended
    extension = starting[-1] + amplitude[-1] - starting[0]
    # normalize everything for fit again in the original dimension
    starting /= extension
    amplitude /= extension
    # bring everything to the original square
    starting = (x if horizontal else y) + starting * (w if horizontal else h)
    amplitude = amplitude * (w if horizontal else h)
    # create each 4-tuple for each new block
    results = [(s, y, a, h) if horizontal else (x, s, w, a)
                for s, a in zip(starting, amplitude)]
    return results


def _reduce_dict(count_dict, partial_key):
    """
    Make partial sum on a counter dict.
    Given a match for the beginning of the category, it will sum each value.
    """
    L = len(partial_key)
    count = sum(v for k, v in count_dict.iteritems() if k[:L] == partial_key)
    return count


def _key_splitting(rect_dict, keys, values, key_subset, horizontal, gap):
    """
    Given a dictionary where each entry  is a rectangle, a list of key and
    value (count of elements in each category) it split each rect accordingly,
    as long as the key start with the tuple key_subset.  The other keys are
    returned without modification.
    """
    result = OrderedDict()
    L = len(key_subset)
    for name, (x, y, w, h) in rect_dict.iteritems():
        if key_subset == name[:L]:
            # split base on the values given
            divisions = _split_rect(x, y, w, h, values, horizontal, gap)
            for key, rect in zip(keys, divisions):
                result[name + (key,)] = rect
        else:
            result[name] = (x, y, w, h)
    return result


def hierarchical_split(count_dict, horizontal=True, gap=0.05):
    """
    Split a square in a hierarchical way given a contingency table.

    Hierarchically split the unit square in alternate directions
    in proportion to the subdivision contained in the contingency table
    count_dict.  This is the function that actually perform the tiling
    for the creation of the mosaic plot.  If the gap array has been specified
    it will insert a corresponding amount of space (proportional to the
    unit lenght), while retaining the proportionality of the tiles.

    Parameters
    ----------
    count_dict : dict
                Dictionary containing the contingency table.
                Each category should contain a non-negative number
                with a tuple as index.  It expects that all the combination
                of keys to be representes; if that is not true, will
                automatically consider the missing values as 0
    horizontal : bool
                The starting direction of the split (by default along
                the horizontal axis)
    gap : float or array of floats
                The list of gaps to be applied on each subdivision.
                If the lenght of the given array is less of the number
                of subcategories (or if it's a single number) it will extend
                it with exponentially decreasing gaps

    Returns
    ----------
    base_rect : dict
                A dictionary containing the result of the split.
                To each key is associated a 4-tuple of coordinates
                that are required to create the corresponding rectangle:
                    0 - x position of the lower left corner
                    1 - y position of the lower left corner
                    2 - width of the rectangle
                    3 - height of the rectangle
    """
    # this is the unit square that we are going to divide
    base_rect = OrderedDict([(tuple(), (0, 0, 1, 1))])
    # use the Ordered dict to implement a simple ordered set
    # return each level of each category
    # [[key_1_level_1,key_2_level_1],[key_1_level_2,key_2_level_2]]
    categories_levels = [list(OrderedDict([(j, None) for j in i]))
                                for i in zip(*(count_dict.keys()))]
    L = len(categories_levels)
    #fill the void in the counting dictionary
    indexes = product(*categories_levels)
    contingency = OrderedDict([(k, count_dict.get(k, 0)) for k in indexes])
    count_dict = contingency
    # recreate the gaps vector starting from an int
    if not numpy.iterable(gap):
        gap = [gap / 1.5 ** idx for idx in range(L)]
    # extend if it's too short
    if len(gap) < L:
        last = gap[-1]
        gap = list(*gap) + [last / 1.5 ** idx for idx in range(L)]
    # trim if it's too long
    gap = gap[:L]
    # put the count dictionay in order for the keys
    # thil will allow some code simplification
    count_ordered = OrderedDict([(k, count_dict[k])
                        for k in list(product(*categories_levels))])
    for cat_idx, cat_enum in enumerate(categories_levels):
        # get the partial key up to the actual level
        base_keys = list(product(*categories_levels[:cat_idx]))
        for key in base_keys:
            # for each partial and each value calculate how many
            # observation we have in the counting dictionary
            part_count = [_reduce_dict(count_ordered, key + (partial,))
                            for partial in cat_enum]
            # reduce the gap for subsequents levels
            new_gap = gap[cat_idx]
            # split the given subkeys in the rectangle dictionary
            base_rect = _key_splitting(base_rect, cat_enum, part_count, key,
                                       horizontal, new_gap)
        horizontal = not horizontal
    return base_rect


def _get_from_partial_key(dict, key, default):
    """match a tuple used as a key to a dict to shorter tuple if not found"""
    while key:
        if key in dict:
            return dict[key]
        key = key[:-1]
    return default


def _single_hsv_to_rgb(hsv):
    """Transform a color from the hsv space to the rgb."""
    from matplotlib.colors import hsv_to_rgb
    return hsv_to_rgb(array(hsv).reshape(1, 1, 3)).reshape(3)


def mosaic(data, ax=None, horizontal=True, gap=0.005,
           properties={}, labelizer=None):
    """
    Create a mosaic plot from a contingency table.

    It allows to visualize multivariate categorical data in a rigorous
    and informative way.  The color scheme can be personalized with the
    properties keyword.

    for more information you can read:
        http://www.math.yorku.ca/SCS/Online/mosaics/about.html
        http://www.theusrus.de/blog/understanding-mosaic-plots/
        http://www.vicc.org/biostatistics/LuncheonTalks/DrTsai2.pdf

    Parameters
    ----------
    data : dict
                The contingency table that contains the data.
                Each category should contain a non-negative number
                with a tuple as index.  It expects that all the combination
                of keys to be representes; if that is not true, will
                automatically consider the missing values as 0
    ax : matplotlib.Axes, optional
                The graph where display the mosaic. If not given, will
                create a new figure
    horizontal : bool
                The starting direction of the split (by default along
                the horizontal axis)
    gap : float or array of floats
                The list of gaps to be applied on each subdivision.
                If the lenght of the given array is less of the number
                of subcategories (or if it's a single number) it will extend
                it with exponentially decreasing gaps
    labelizer : lambda (key,data) -> string, optional
                A lambda function that generate the text to display
                at the center of each tile base on the dataset and the
                key related to that tile
    properties : dict
                Contains the properties for each tile, using the same
                key as the dataset as index.  The properties are used to
                create a matplotlib.Rectangle.  If the key is not found it
                search for a partial submatch (a more general category).
                For a general value on all categories unless specified use a
                default dict with the chose aspect.

    Returns
    ----------
    fig : matplotlib.Figure
                The generate figure
    rects : dict
                A dictionary that has the same keys of the original
                dataset, that holds a reference to the coordinates of the
                tile and the Rectangle that represent it
    """
    from pylab import Rectangle
    fig, ax = utils.create_mpl_ax(ax)
    rects = hierarchical_split(data, horizontal=horizontal, gap=gap)
    if labelizer is None:
        labelizer = lambda k: "\n".join(k) + "\ncount=" + str(data.get(k, 0))
    for k, v in rects.items():
        x, y, w, h = v
        conf = _get_from_partial_key(properties, k, {})
        Rect = Rectangle((x, y), w, h, **conf)
        test = labelizer(k)
        ax.add_patch(Rect)
        ax.text(x + w / 2, y + h / 2, test, ha='center',
                 va='center', size='smaller')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    return fig, rects
