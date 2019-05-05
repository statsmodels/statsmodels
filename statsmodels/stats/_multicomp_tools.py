# -*- coding: utf-8 -*-
"""
Created on Tue May 16 12:27:57 2017

Author: Josef Perktold
License: BSD-3

"""

import numpy as np


class GroupingResults(object):

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __str__(self):
        # TODO: need to create a table with aligned columns
        slen = max(len(name) for name in self.names)
        #sformat = '\%' + str(slen)
        s = '\n'.join(name + ": " + " "*(slen - len(name)) + lett
                      for name, lett in zip(self.names, self.letter))
        return s


def to_remove(set_, distance):
    for i in set_:
        if not set_ <= frozenset(np.nonzero(distance[i])[0]):
            return True

    return False


def group_pairwise(edges, distance_matrix=None, k_groups=None, names=None,
                   maxiter=100):
    """form indifference groups after pairwise comparison

    This function generate possibly overlapping groups for fully
    connected sets in a graph.

    Parameters
    ----------
    edges : array_like, 2-D, or None
        array or list of lists with edges in rows and two.
        edges can be None if a distance_matrix is provided
    distance_matrix : None, or symmetric array

    k_groups : None or int
        number of groups in pairwise comparison
        If None, then it is inferred from edges or distance_matrix
    names : None or list of strings

    maxiter : int
        number of iteration in the intersection step

    Returns
    -------
    results : GroupingResults instance
        The main attributes are
        - groupings : list of frozensets
        - letter : list of strings, letter assignment to groupings

    Notes
    -----
    The grouping of pairwise significance does not assume ordered boundaries
    as in Piepho 2004. Piepho describes it as letter display which is a more
    general version of a line display. This letter display is also valid
    when groups in the pairwise comparison have unequal variances and
    significance depends on both means and variances, while line displays
    assume that significance is ordered by the means.

    However, this function uses a different algorithm than Piepho that
    takes advantage of Python sets. (There is no reference for the
    algorithm used here but I expect it to be faster than Piepho's algorithm
    if there are many disconnected groupings.)
    Graph and network libraries will have better algorithms for large graphs.

    Groupings in the Results are sorted by the smallest member of a group to
    avoid indeterminate ordering and letter assignments given by sets.
    Consequently the assigned letters will also differs from Piepho 2004.
    This also does not use a sweeping step.

    Status : API is very experimental and will still change. This still needs
        support for pandas indexes. Currently insufficient input verification.

    References
    ----------
    Piepho, Hans-Peter. 2004. “An Algorithm for a Letter-Based Representation
        of All-Pairwise Comparisons.”
        Journal of Computational and Graphical Statistics 13 (2): 456–66.


    """
    if distance_matrix is None:
        edges = np.asarray(edges)
        if k_groups is None:
            k_groups = edges.max() + 1
        d = np.zeros((k_groups, k_groups))
        d[edges[:, 0], edges[:,1]] = 1
        d += d.T
    else:
        d = np.asarray(distance_matrix)
        k_groups = d.shape[0]

    if names is None:
        names = ["g_%d" % i for i in range(k_groups)]

    d = 1 - d
    allsets = set(frozenset(np.nonzero(d[i])[0]) for i in range(k_groups))
    # add intersections
    allsets_old = allsets
    allsets_new = set()
    for ii in range(maxiter):
        allsets_new = set(set_.intersection(frozenset(np.nonzero(d[i])[0]))
                     for set_ in allsets_old for i in set_)
        if allsets_new == allsets_old:
            converged = True
            break
        allsets_old = allsets_new
    else:
        converged = False

    allsets = allsets_new
    # remove incorrect sets, sets including a significant difference
    # we switch to list to get ordered sequence
    all_ = []
    for s in allsets:
        if not to_remove(s, d):
            all_.append(s)

    # sort by smallest element to get deterministic representation
    # sort by len first to break ties in min
    # no sorting if both len and min are equal for several groupings
    all_.sort(key=len, reverse=True)
    all_.sort(key=min)

    # assigning letters
    alphabet = 'abcdefghijklmnop'
    alphabet += alphabet.upper()
    if len(alphabet) < len(all_):
        raise ValueError("Not enough letters in alphabet for groupings")
    k_letters = len(all_)
    letter = [] * k_letters
    for i in range(k_groups):
        ss = ''.join(alphabet[j] if i in set_j else ' ' for j, set_j in enumerate(all_))
        letter.append(ss)

    res =  GroupingResults(distance_matrix=d,
                           groupings=all_,
                           letter=letter,
                           k_groups=k_groups,
                           allsets=allsets,
                           names=names,
                           iterations=ii,
                           converged=converged)

    return res
