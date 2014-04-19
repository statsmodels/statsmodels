# -*- coding: utf-8 -*-
"""
Halton sequence

Original source:
Quasi Montecarlo Halton sequence generator
Created on Mon Jun 17 22:12:21 2013
Author: Sebastien Paris
Josef Perktold translation from c

http://www.mathworks.com/matlabcentral/fileexchange/17457-quasi-montecarlo-halton-sequence-generator
http://jpktd.blogspot.ca/2013/06/quasi-random-numbers-with-halton.html
"""

#void halton(int dim , int nbpts, double *h  , double *p )
#{
#
#	double lognbpts , d , sum;
#
#	int i , j , n , t , b;
#
#	static int P[11] = {2 ,3 ,5 , 7 , 11 , 13 , 17 , 19 ,  23 , 29 , 31};
#
#
#	lognbpts = log(nbpts + 1);
#
#
#	for(i = 0 ; i < dim ; i++)
#
#	{
#
#		b      = P[i];
#
#		n      = (int) ceil(lognbpts/log(b));
#
#
#		for(t = 0 ; t < n ; t++)
#
#		{
#			p[t] = pow(b , -(t + 1) );
#		}
#
#
#		for (j = 0 ; j < nbpts ; j++)
#
#		{
#
#			d        = j + 1;
#
#			sum      = fmod(d , b)*p[0];
#
#
#			for (t = 1 ; t < n ; t++)
#
#			{
#
#				d        = floor(d/b);
#
#				sum     += fmod(d , b)*p[t];
#
#			}
#
#
#			h[j*dim + i] = sum;
#
#		}
#
#	}
#
#}

from math import log, floor, ceil, fmod
import numpy as np

def halton(dim, nbpts):
    """
    Halton sequence

    Parameters
    ----------
    dim : float
         width of sequence
    nbpts : float
        length of sequence

    Returns
    -------
    halton : array (nbpts * dim)
        halton sequence

    Notes
    ------
    Quasi Montecarlo generator
    primes numbers set to:
    P = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    """

    h = np.empty(nbpts * dim)
    h.fill(np.nan)
    p = np.empty(nbpts)
    p.fill(np.nan)

    # TODO: provided prime number instead of set it static
    P = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,\
         59, 61, 67, 71, 73, 79, 83, 89]
    lognbpts = log(nbpts + 1)
    for i in range(dim):
        b = P[i]
        n = int(ceil(lognbpts / log(b)))
        for t in range(n):
            p[t] = pow(b, -(t + 1) )

        for j in range(nbpts):
            d = j + 1
            sum_ = fmod(d, b) * p[0]
            for t in range(1, n):
                d = floor(d / b)
                sum_ += fmod(d, b) * p[t]

            h[j*dim + i] = sum_

    return h.reshape(nbpts, dim)

if __name__ == "__main__":

    x = halton(2, 500)
    #plot(x(1 , :) , x(2 , :) , '+')
    print(x[:5])
    xr = np.random.rand(500, 2)

    import matplotlib.pyplot as plt
    fig1 = plt.figure()
    ax = fig1.add_subplot(2,2,1)
    ax.plot(x[:500, 0], x[:500, 1], '+')
    ax.set_title('uniform-distribution (Halton)')

    ax = fig1.add_subplot(2,2,3)
    ax.plot(xr[:500, 0], xr[:500, 1], '+')
    ax.set_title('uniform-distribution (random)')


    #plt.figure()
    #plt.plot(x[:, 0], x[:, 1], '+')
    #plt.title('uniform-distribution')

    from scipy import stats

    xn0 = stats.norm._ppf(x)
    trans = np.linalg.cholesky(np.linalg.inv([[1, 0.7], [0.7, 1]]))
    trans = np.linalg.cholesky([[1, 0.7], [0.7, 1]])
    xn = np.dot(xn0, trans.T)
    ax = fig1.add_subplot(2,2,2)
    ax.plot(xn[:, 0], xn[:, 1], '+')
    ax.set_title('normal-distribution (Halton)')

    xrn = stats.norm._ppf(xr)
    xrn = np.dot(xrn, trans)
    ax = fig1.add_subplot(2,2,4)
    ax.plot(xrn[:, 0], xrn[:, 1], '+')
    ax.set_title('normal-distribution (random)')

    fig2 = plt.figure()
    xln = np.exp(xn)
    ax = fig2.add_subplot(2,2,1)
    plt.plot(xln[:, 0], xln[:, 1], '+')
    plt.title('log-normal-distribution (Halton)')

    ax = fig2.add_subplot(2,2,2)
    ax.plot(stats.t._ppf(x[:, 0], 3), stats.t._ppf(x[:, 1], 3), '+')
    ax.set_title('t-distribution (Halton)')


    x0 = xn[:100]
    x0 /= np.sqrt((x0*x0 + 1e-100).sum(1))[:,None]
    ax = fig2.add_subplot(2,2,3)
    ax.plot(x0[:, 0] , x0[:, 1], '+')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title('normal projected on circle')

    xlarge = halton(2, 5000)
    ax = fig2.add_subplot(2,2,4)
    ax.plot(xlarge[:, 0], xlarge[:, 1], '+')
    ax.set_title('uniform-distribution (Halton, 2000 points)')


    plt.show()
