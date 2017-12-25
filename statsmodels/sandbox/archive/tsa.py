'''Collection of alternative implementations for time series analysis

'''


'''
>>> signal.fftconvolve(x,x[::-1])[len(x)-1:len(x)+10]/x.shape[0]
array([  2.12286549e+00,   1.27450889e+00,   7.86898619e-02,
        -5.80017553e-01,  -5.74814915e-01,  -2.28006995e-01,
         9.39554926e-02,   2.00610244e-01,   1.32239575e-01,
         1.24504352e-03,  -8.81846018e-02])
>>> sm.tsa.stattools.acovf(X, fft=True)[:order+1]
array([  2.12286549e+00,   1.27450889e+00,   7.86898619e-02,
        -5.80017553e-01,  -5.74814915e-01,  -2.28006995e-01,
         9.39554926e-02,   2.00610244e-01,   1.32239575e-01,
         1.24504352e-03,  -8.81846018e-02])

>>> import nitime.utils as ut
>>> ut.autocov(s)[:order+1]
array([  2.12286549e+00,   1.27450889e+00,   7.86898619e-02,
        -5.80017553e-01,  -5.74814915e-01,  -2.28006995e-01,
         9.39554926e-02,   2.00610244e-01,   1.32239575e-01,
         1.24504352e-03,  -8.81846018e-02])
'''

from statsmodels.tsa.stattools import acovf

def acovf_fft(x, demean=True):
    """
    acovf_fft is deprecated, will be removed [...].  It will continue
    working until then.  Users are encouraged to
    use sm.tsa.stattools.acovf instead.
    """
    return acovf(x, unbiased=False, demean=demean, fft=True, missing='none')

