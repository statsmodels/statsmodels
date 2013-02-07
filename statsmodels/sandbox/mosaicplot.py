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


def _tuplify(obj):
    return tuple(obj) if numpy.iterable(obj) else (obj,)


def _categories_level(keys):
    """use the Ordered dict to implement a simple ordered set
    return each level of each category
    [[key_1_level_1,key_2_level_1],[key_1_level_2,key_2_level_2]]"""
    res = []
    for i in zip(*(keys)):
        tuplefied = _tuplify(i)
        res.append(list(OrderedDict([(j, None) for j in tuplefied])))
    return res
    #return [list(OrderedDict([(j, None) for j in tuple(i)]))
    #                                        for i in zip(*(keys))]

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
    labels : dict
                the center of the labels for each subdivision
    """
    # this is the unit square that we are going to divide
    base_rect = OrderedDict([(tuple(), (0, 0, 1, 1))])
    #get the list of each possible value for each level
    categories_levels = _categories_level(count_dict.keys())
    L = len(categories_levels)

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
    # this will allow some code simplification
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


def _create_default_properties(data):
    """"create the default properties of the mosaic given the data"""
    categories_levels = _categories_level(data.keys())
    Nlevels = len(categories_levels)
    #first level, the hue
    L = len(categories_levels[0])
    #hue = numpy.linspace(1.0, 0.0, L+1)[:-1]
    hue = numpy.linspace(0.0, 1.0, L + 2)[:-2]
    #second level, the saturation
    L = len(categories_levels[1]) if Nlevels > 1 else 1
    saturation = numpy.linspace(0.5, 1.0, L + 1)[:-1]
    #third level, the value
    L = len(categories_levels[2]) if Nlevels > 2 else 1
    value = numpy.linspace(0.5, 1.0, L + 1)[:-1]
    #fourth level, the hatch
    L = len(categories_levels[3]) if Nlevels > 3 else 1
    hatch = ['', '/', '-', '|', '+'][:L + 1]
    #convert in list and merge with the levels
    hue = zip(list(hue), categories_levels[0])
    saturation = zip(list(saturation),
                     categories_levels[1] if Nlevels > 1 else [''])
    value = zip(list(value),
                     categories_levels[2] if Nlevels > 2 else [''])
    hatch = zip(list(hatch),
                     categories_levels[3] if Nlevels > 3 else [''])
    #create the properties dictionary
    properties = {}
    for h, s, v, t in product(hue, saturation, value, hatch):
        hv, hn = h
        sv, sn = s
        vv, vn = v
        tv, tn = t
        level = (hn,) + ((sn,) if sn else tuple())
        level = level + ((vn,) if vn else tuple())
        level = level + ((tn,) if tn else tuple())
        hsv = array([hv, sv, vv])
        prop = {'color': _single_hsv_to_rgb(hsv), 'hatch': tv}
        properties[level] = prop
    return properties


#def _label_position(split_rect, horizontal = True):
#    """find the label position for each category"""
#    keys = split_rect.keys()
#    categories_levels = _categories_level(keys)
#    positions = {}
#    for level in range(len(categories_levels)):
#        names = categories_levels[level]
#        # on which side the label will be on
#        orientation = (level + int(not horizontal)) % 4
#        for name in names:
#            positions[name] = ([],orientation)
#            for key in keys:
#                if key[level] == name:
#                    x, y, w, h = split_rect[key]
#                    pos = [x + w / 2, y + h / 2]
#                    if orientation == 0:
#                        pos[1] = 0.0
#                    if orientation == 2:
#                        pos[1] = 1.0
#                    if orientation == 1:
#                        pos[0] = 0.0
#                    if orientation == 3:
#                        pos[0] = 1.0
#                    positions[name][0].append(pos)
#                    if (orientation // 2):
#                        break
#    return positions


def _normalize_data(data):
    ##TODO: complete the normalization function and use it
    """normalize the data to a dict with tuples as keys
    right now it works with:
        dictionary with simple keys
        pandas.Series with simple or hierarchical indexes
    to be implemented:
        numpy arrays (need info on the name sequence)"""
    try:
        items = data.iteritems()
    except AttributeError:
        #ok, I cannot use the data as a dictionary
        #it may be a list of an array and the sequence of names
        #of the levels
        if isinstance(data, list) and len(data) == 2:
            data, names = data
            temp = {}
            for idx in numpy.ndindex(data.shape):
                name = tuple(names[n][i] for n, i in enumerate(idx))
                temp[name] = data[idx]
            data = temp
            items = data.iteritems()
        #or it may be a simple array without labels
        elif isinstance(data, numpy.ndarray):
            temp = {}
            for idx in numpy.ndindex(data.shape):
                name = tuple(str(i) for i in idx)
                temp[name] = data[idx]
            data = temp
            items = data.iteritems()
    data = OrderedDict([_tuplify(k), v] for k, v in items)
    categories_levels = _categories_level(data.keys())
    # fill the void in the counting dictionary
    indexes = product(*categories_levels)
    contingency = OrderedDict([(k, data.get(k, 0)) for k in indexes])
    data = contingency
    return data


def mosaic(data, ax=None, horizontal=True, gap=0.005,
           properties={}, labelizer=None, title = ''):
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
    data : dict, pandas.Series, numpy.ndarray, [numpy.ndarray, labels]
        The contingency table that contains the data.
        Each category should contain a non-negative number
        with a tuple as index.  It expects that all the combination
        of keys to be representes; if that is not true, will
        automatically consider the missing values as 0.  The order
        of the keys will be the same as the one of insertion.
        If a dict of a Series (or any other dict like object)
        is used, it will take the keys as labels.  If a
        numpy.ndarray is provided, it will generate a simple
        numerical labels. A tuple (or list) containing the
        ndarray with the list of labels can be provided.  In this
        Case the label should be a list of list, where each
        sublist should be the name of each category for each level.
    ax : matplotlib.Axes, optional
        The graph where display the mosaic. If not given, will
        create a new figure
    horizontal : bool, optional (default True)
        The starting direction of the split (by default along
        the horizontal axis)
    gap : float or array of floats
        The list of gaps to be applied on each subdivision.
        If the lenght of the given array is less of the number
        of subcategories (or if it's a single number) it will extend
        it with exponentially decreasing gaps
    labelizer : function (key) -> string, optional
        A lambda function that generate the text to display
        at the center of each tile base on the dataset and the
        key related to that tile
    properties : dict, optional
        Contains the properties for each tile, using the same
        key as the dataset as index.  The properties are used to
        create a matplotlib.Rectangle.  If the key is not found it
        search for a partial submatch (a more general category).
        For a general value on all categories unless specified use a
        default dict with the chose aspect.
    title : string, optional
        The title of the axis

    Returns
    ----------
    fig : matplotlib.Figure
                The generate figure
    rects : dict
                A dictionary that has the same keys of the original
                dataset, that holds a reference to the coordinates of the
                tile and the Rectangle that represent it

    Examples
    --------
    The most simple use case is to take a dictionary and plot the result

    >>> data = {'a': 10, 'b': 15, 'c': 16}
    >>> mosaic(data, title='basic dictionary')
    >>> pylab.show()

    A more useful example is given by a dictionary with multiple indices.
    In this case we use a wider gap to a better visual separation of the
    resulting plot

    >>> data = {('a', 'b'): 1, ('a', 'c'): 2, ('d', 'b'): 3, ('d', 'c'): 4}
    >>> mosaic(data, gap=0.05, title='complete dictionary')
    >>> pylab.show()

    The same data can be given as a simple or hierarchical indexed Series

    >>> rand = numpy.random.random
    >>> from itertools import product
    >>>
    >>> tuples = list(product(['bar', 'baz', 'foo', 'qux'], ['one', 'two']))
    >>> index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
    >>> data = pd.Series(rand(8), index=index)
    >>> mosaic(data, title='hierarchical index series')
    >>> pylab.show()

    The third accepted data structureis the numpy array, for which a
    very simple index will be created.  It is possible to pass the index in
    an explicit way

    >>> rand = numpy.random.random
    >>> data = 1+rand((2,2))
    >>> mosaic(data, title='random non-labeled array')
    >>> pylab.show()

    >>> labels = [['first', 'second'], ['foo', 'spam']]
    >>> mosaic([data, labels], title='random labeled array')
    >>> pylab.show()

    """
    from pylab import Rectangle
    fig, ax = utils.create_mpl_ax(ax)
    # create a dictionary with only tuplified keys
    #items = data.iteritems()
    #data = OrderedDict([_tuplify(k), v] for k, v in items)
    data = _normalize_data(data)
    # split the graph into different areas
    rects = hierarchical_split(data, horizontal=horizontal, gap=gap)
    #for label, ((x, y), side) in labels.iteritems():
    #    ax.text(x, y, label, ha='center', va='center', size='smaller')
    if labelizer is None:
        labelizer = lambda k: "\n".join(k)
    default_props = _create_default_properties(data)
    for k, v in rects.items():
        x, y, w, h = v
        conf = _get_from_partial_key(properties, k, {})
        props = conf if conf else default_props[k]
        text = labelizer(k)
        Rect = Rectangle((x, y), w, h, label=text, lw=0, **props)
        ax.add_patch(Rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center',
                 va='center', size='smaller')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_title(title)
    return fig, rects


if __name__ == '__main__':
    import matplotlib.pyplot as pylab
    import pandas as pd

    data = {'a': 10, 'b': 15, 'c': 16}
    mosaic(data)
    pylab.show()

    fig, ax = pylab.subplots(3, 3)

    data = {'a': 10, 'b': 15, 'c': 16}
    mosaic(data, gap=0.05, title='basic dictionary', ax=ax[0, 0])

    data = pd.Series(data)
    mosaic(data, gap=0.05, title='basic series', ax=ax[0, 1])

    data = {('a', 'b'): 1, ('a', 'c'): 2, ('d', 'b'): 3, ('d', 'c'): 4}
    mosaic(data, gap=0.05, title='complete dictionary', ax=ax[1, 0])

    data = pd.Series(data)
    mosaic(data, gap=0.05, title='complete series', ax=ax[1, 1])

    data = {('a', 'b'): 1, ('a', 'c'): 2, ('d', 'b'): 3}
    mosaic(data, gap=0.05, title='incomplete dictionary', ax=ax[1, 2])

    # creation of the levels
    key_set = [['male', 'female'], ['old', 'adult', 'young'],
               ['worker', 'unemployed'], ['healty', 'ill']]
    keys = list(product(*key_set))
    data = OrderedDict(zip(keys, range(1, 1 + len(keys))))
    #use a function to change hiw the label are shown
    labelizer = lambda k: "".join(n[0].upper() for n in k)
    mosaic(data, gap=0.05, title='complex dictionary + labelization',
        ax=ax[0, 2], labelizer=labelizer)

    rand = numpy.random.random
    data = 1+rand((2,2))
    mosaic(data, gap=0.05, title='random non-labeled array', ax=ax[2, 0])

    labels = [['first', 'second'], ['foo', 'spam']]
    mosaic([data, labels], gap=0.05, title='random labeled array', ax=ax[2, 1])

    from itertools import product
    tuples = list(product(['bar', 'baz', 'foo', 'qux'], ['one', 'two']))
    index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
    data = pd.Series(rand(8), index=index)
    mosaic(data, gap=0.005, title='hierarchical index series', ax=ax[2, 2])

    pylab.show()