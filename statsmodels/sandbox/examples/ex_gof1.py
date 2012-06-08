# -*- coding: utf-8 -*-
"""

Created on Fri Jun 08 16:13:47 2012

Author: Josef Perktold
"""



if __name__ == '__main__':
    from scipy import stats
    #rvs = np.random.randn(1000)
    rvs = stats.t.rvs(3, size=200)
    print 'scipy kstest'
    print kstest(rvs, 'norm')
    goft = GOF(rvs, 'norm')
    print goft.get_test()

    all_gofs = ['d', 'd_plus', 'd_minus', 'v', 'wsqu', 'usqu', 'a']
    for ti in all_gofs:
        print ti, goft.get_test(ti, 'stephens70upp')

    print '\nIs it correctly sized?'
    from collections import defaultdict

    results = defaultdict(list)
    nobs = 200
    for i in xrange(100):
        rvs = np.random.randn(nobs)
        goft = GOF(rvs, 'norm')
        for ti in all_gofs:
            results[ti].append(goft.get_test(ti, 'stephens70upp')[0][1])

    resarr = np.array([results[ti] for ti in all_gofs])
    print '         ', '      '.join(all_gofs)
    print 'at 0.01:', (resarr < 0.01).mean(1)
    print 'at 0.05:', (resarr < 0.05).mean(1)
    print 'at 0.10:', (resarr < 0.1).mean(1)

    gof_mc(lambda nobs: stats.t.rvs(3, size=nobs), 'norm', nobs=200)

    nobs = 200
    nrep = 100
    bt = bootstrap(NewNorm(), args=(0,1), nobs=nobs, nrep=nrep, value=None)
    quantindex = np.floor(nrep * np.array([0.99, 0.95, 0.9])).astype(int)
    print bt[quantindex]

    #the bootstrap results match Stephens pretty well for nobs=100, but not so well for
    #large (1000) or small (20) nobs
    '''
    >>> np.array([15.0, 10.0, 5.0, 2.5, 1.0])/100.  #Stephens
    array([ 0.15 ,  0.1  ,  0.05 ,  0.025,  0.01 ])
    >>> nobs = 100
    >>> [bootstrap(NewNorm(), args=(0,1), nobs=nobs, nrep=10000, value=c/ (1 + 4./nobs - 25./nobs**2)) for c in [0.576, 0.656, 0.787, 0.918, 1.092]]
    [0.1545, 0.10009999999999999, 0.049000000000000002, 0.023, 0.0104]
    >>>
    '''

    #test equality of loop, vectorized, batch-vectorized
    np.random.seed(8765679)
    resu1 = bootstrap(NewNorm(), args=(0,1), nobs=nobs, nrep=100,
                      value=0.576/(1 + 4./nobs - 25./nobs**2))
    np.random.seed(8765679)
    tmp = [bootstrap(NewNorm(), args=(0,1), nobs=nobs, nrep=1) for _ in range(100)]
    resu2 = (np.array(tmp) > 0.576/(1 + 4./nobs - 25./nobs**2)).mean()
    np.random.seed(8765679)
    tmp = [bootstrap(NewNorm(), args=(0,1), nobs=nobs, nrep=1,
                     value=0.576/ (1 + 4./nobs - 25./nobs**2),
                     batch_size=10) for _ in range(10)]
    resu3 = np.array(tmp).mean()
    from numpy.testing import assert_almost_equal, assert_array_almost_equal
    assert_array_almost_equal(resu1, resu2, 15)
    assert_array_almost_equal(resu2, resu3, 15)
