
if __name__ == '__main__':

    # ---------- examples combine_indices
    from numpy.testing import assert_equal

    np.random.seed(985367)
    groups = np.random.randint(0, 2, size=(10, 2))
    uv, ux, u, label = combine_indices(groups, return_labels=True)
    uv, ux, u, label = combine_indices(groups, prefix='g1,g2=', sep=',',
                                       return_labels=True)

    group0 = np.array(['sector0', 'sector1'])[groups[:, 0]]
    group1 = np.array(['region0', 'region1'])[groups[:, 1]]
    uv, ux, u, label = combine_indices((group0, group1),
                                       prefix='sector,region=',
                                       sep=',',
                                       return_labels=True)
    uv, ux, u, label = combine_indices((group0, group1), prefix='', sep='.',
                                       return_labels=True)
    group_joint = np.array(label)[uv]
    group_joint_expected = np.array(['sector1.region0', 'sector0.region1',
                                     'sector0.region0', 'sector0.region1',
                                     'sector1.region1', 'sector0.region0',
                                     'sector1.region0', 'sector1.region0',
                                     'sector0.region1', 'sector0.region0'],
                                    dtype='|S15')
    assert_equal(group_joint, group_joint_expected)

    """
    >>> uv
    array([2, 1, 0, 0, 1, 0, 2, 0, 1, 0])
    >>> label
    ['sector0.region0', 'sector1.region0', 'sector1.region1']
    >>> np.array(label)[uv]
    array(['sector1.region1', 'sector1.region0', 'sector0.region0',
           'sector0.region0', 'sector1.region0', 'sector0.region0',
           'sector1.region1', 'sector0.region0', 'sector1.region0',
           'sector0.region0'],
          dtype='|S15')
    >>> np.column_stack((group0, group1))
    array([['sector1', 'region1'],
           ['sector1', 'region0'],
           ['sector0', 'region0'],
           ['sector0', 'region0'],
           ['sector1', 'region0'],
           ['sector0', 'region0'],
           ['sector1', 'region1'],
           ['sector0', 'region0'],
           ['sector1', 'region0'],
           ['sector0', 'region0']],
          dtype='|S7')
      """

    # ------------- examples sparse_dummies
    from scipy import sparse

    g = np.array([0, 0, 1, 2, 1, 1, 2, 0])
    u = lrange(3)
    indptr = np.arange(len(g)+1)
    data = np.ones(len(g), dtype=np.int8)
    a = sparse.csr_matrix((data, g, indptr))
    print(a.todense())
    print(np.all(a.todense() == (g[:, None] == np.arange(3)).astype(int)))

    x = np.arange(len(g)*3).reshape(len(g), 3, order='F')

    print('group means')
    print(x.T * a)
    print(np.dot(x.T, g[:, None] == np.arange(3)))
    print(np.array([np.bincount(g, weights=x[:, col]) for col in range(3)]))
    for cat in u:
        print(x[g == cat].sum(0))
    for cat in u:
        x[g == cat].sum(0)

    cc = sparse.csr_matrix([[0, 1, 0, 1, 0, 0, 0, 0, 0],
                            [1, 0, 1, 0, 1, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 1, 0, 0, 0],
                            [1, 0, 0, 0, 1, 0, 1, 0, 0],
                            [0, 1, 0, 1, 0, 1, 0, 1, 0],
                            [0, 0, 1, 0, 1, 0, 0, 0, 1],
                            [0, 0, 0, 1, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 1, 0, 1, 0, 1],
                            [0, 0, 0, 0, 0, 1, 0, 1, 0]])

    # ------------- groupsums
    print(group_sums(np.arange(len(g)*3*2).reshape(len(g), 3, 2), g,
                     use_bincount=False).T)
    print(group_sums(np.arange(len(g)*3*2).reshape(len(g), 3, 2)[:, :, 0], g))
    print(group_sums(np.arange(len(g)*3*2).reshape(len(g), 3, 2)[:, :, 1], g))

    # ------------- examples class
    x = np.arange(len(g)*3).reshape(len(g), 3, order='F')
    mygroup = Group(g)
    print(mygroup.group_int)
    print(mygroup.group_sums(x))
    print(mygroup.labels())