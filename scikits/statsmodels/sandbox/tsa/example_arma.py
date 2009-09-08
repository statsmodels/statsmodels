'''trying to verify theoretical acf of arma

'''

import numpy as np
from numpy.testing import assert_array_almost_equal

from arima import arma_generate_sample, arma_impulse_response
from arima import arma_acovf, arma_acf
from movstat import acf, acovf

ar = [1., -0.6]
#ar = [1., 0.]
ma = [1., 0.4]
#ma = [1., 0.4, 0.6]
#ma = [1., 0.]
mod = ''#'ma2'
x = arma_generate_sample(ar, ma, 5000)
x_acf = acf(x)[:10]
x_ir = arma_impulse_response(ar, ma)

#moved to arima
#def arma_acovf(ar, ma, nobs=10):
#    '''theoretical autocovariance function of ARMA process
#    '''
#    ir = arma_impulse_response(ar, ma)
#    acovf = np.correlate(ir,ir,'full')[len(ir)-1:]
#    return acovf[:10]
#
#def arma_acf(ar, ma, nobs=10):
#    '''theoretical autocovariance function of ARMA process
#    '''
#    acovf = arma_acovf(ar, ma, nobs)
#    return acovf/acovf[0]

#print x_acf[:10]
#print x_ir[:10]
#irc2 = np.correlate(x_ir,x_ir,'full')[len(x_ir)-1:]
#print irc2[:10]
#print irc2[:10]/irc2[0]
#print irc2[:10-1] / irc2[1:10]
#print x_acf[:10-1] / x_acf[1:10]

def acovf_explicit(ar, ma, nobs):
    '''add correlation of MA representation explicitely

    '''
    ir = arma_impulse_response(ar, ma)
    acovfexpl = [np.dot(ir[:nobs-t], ir[t:nobs]) for t in range(10)]
    return acovfexpl

def acovf_arma11(ar, ma):
    # ARMA(1,1)
    # Florens et al page 278
    # wrong result ?
    # new calculation bigJudge p 311, now the same
    a = -ar[1]
    b = ma[1]
    #rho = [1.]
    #rho.append((1-a*b)*(a-b)/(1.+a**2-2*a*b))
    rho = [(1.+b**2+2*a*b)/(1.-a**2)]
    rho.append((1+a*b)*(a+b)/(1.-a**2))
    for _ in range(8):
        last = rho[-1]
        rho.append(a*last)
    return np.array(rho)

#    print acf11[:10]
#    print acf11[:10] /acf11[0]

def acovf_ma2(ma):
    # MA(2)
    # from Greene p616 (with typo), Florens p280
    b1 = -ma[1]
    b2 = -ma[2]
    rho = np.zeros(10)
    rho[0] = (1 + b1**2 + b2**2)
    rho[1] = (-b1 + b1*b2)
    rho[2] = -b2
    return rho

#    rho2 = rho/rho[0]
#    print rho2
#    print irc2[:10]/irc2[0]

def acovf_ma1(ma):
    # MA(1)
    # from Greene p616 (with typo), Florens p280
    b = -ma[1]
    rho = np.zeros(10)
    rho[0] = (1 + b**2)
    rho[1] = -b
    return rho

#    rho2 = rho/rho[0]
#    print rho2
#    print irc2[:10]/irc2[0]


ar1 = [1., -0.99]
ar0 = [1., 0.]
ma1 = [1., 0.4]
ma2 = [1., 0.4, 0.6]
ma0 = [1., 0.]

comparefn = dict(
        [('ma1', acovf_ma1),
        ('ma2', acovf_ma2),
        ('arma11', acovf_arma11),
        ('ar1', acovf_arma11)])

cases = [('ma1', (ar0, ma1)),
        ('ma2', (ar0, ma2)),
        ('arma11', (ar1, ma1)),
        ('ar1', (ar1, ma0))]

for c, args in cases:

    ar, ma = args
    print
    print c, ar, ma
    myacovf = arma_acovf(ar, ma, nobs=10)
    myacf = arma_acf(ar, ma, nobs=10)
    if c[:2]=='ma':
        othacovf = comparefn[c](ma)
    else:
        othacovf = comparefn[c](ar, ma)
    print myacovf[:5]
    print othacovf[:5]
    assert_array_almost_equal(myacovf, othacovf,10)
    assert_array_almost_equal(myacf, othacovf/othacovf[0],10)
