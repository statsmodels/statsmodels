from __future__ import print_function
from time import time
from statsmodels.tsa.arma_mle import Arma
from statsmodels.tsa.api import ARMA
import numpy as np

print("Battle of the dueling ARMAs")

y_arma22 = np.loadtxt(r'C:\Josef\eclipsegworkspace\statsmodels-josef-experimental-gsoc\scikits\statsmodels\tsa\y_arma22.txt')

arma1 = Arma(y_arma22)
arma2 = ARMA(y_arma22)

print("The actual results from gretl exact mle are")
params_mle = np.array([.826990, -.333986, .0362419, -.792825])
sigma_mle = 1.094011
llf_mle = -1510.233
print("params: ", params_mle)
print("sigma: ", sigma_mle)
print("llf: ", llf_mle)
print("The actual results from gretl css are")
params_css = np.array([.824810, -.337077, .0407222, -.789792])
sigma_css = 1.095688
llf_css = -1507.301

results = []
results += ["gretl exact mle", params_mle, sigma_mle, llf_mle]
results += ["gretl css", params_css, sigma_css, llf_css]

t0 = time()
print("Exact MLE - Kalman filter version using l_bfgs_b")
arma2.fit(order=(2,2), trend='nc')
t1 = time()
print("params: ", arma2.params)
print("sigma: ", arma2.sigma2**.5)
arma2.llf = arma2.loglike(arma2._invtransparams(arma2.params))
results += ["exact mle kalmanf", arma2.params, arma2.sigma2**.5, arma2.llf]
print('time used:', t1-t0)

t1=time()
print("CSS MLE - ARMA Class")
arma2.fit(order=(2,2), trend='nc', method="css")
t2=time()
arma2.llf = arma2.loglike_css(arma2._invtransparams(arma2.params))
print("params: ", arma2.params)
print("sigma: ", arma2.sigma2**.5)
results += ["css kalmanf", arma2.params, arma2.sigma2**.5, arma2.llf]
print('time used:', t2-t1)

print("Arma.fit_mle results")
# have to set nar and nma manually
arma1.nar = 2
arma1.nma = 2
t2=time()
ret = arma1.fit_mle()
t3=time()
print("params, first 4, sigma, last 1 ", ret.params)
results += ["Arma.fit_mle ", ret.params[:4], ret.params[-1], ret.llf]
print('time used:', t3-t2)

print("Arma.fit method = \"ls\"")
t3=time()
ret2 = arma1.fit(order=(2,0,2), method="ls")
t4=time()
print(ret2[0])
results += ["Arma.fit ls", ret2[0]]
print('time used:', t4-t3)

print("Arma.fit method = \"CLS\"")
t4=time()
ret3 = arma1.fit(order=(2,0,2), method="None")
t5=time()
print(ret3)
results += ["Arma.fit other", ret3[0]]
print('time used:', t5-t4)

for i in results: print(i)

