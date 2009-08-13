import numpy as np
from models.tools import add_constant
from models.regression import AR, yule_walker

X = np.arange(1,8)
X = add_constant(X)
Y = np.array((1, 3, 4, 5, 8, 10, 9))
rho = 2
model = AR(Y, X, 2)
for i in range(6):
    results = model.fit()
    print "AR coefficients:", model.rho
    rho, sigma = yule_walker(results.resid, order = model.order)
    model = AR(Y, X, rho)
par0 = results.params
print par0
model0if = AR(Y, X, 2)
res = model0if.iterative_fit(6)
print 'iterativefit beta', res.params
results.t() # is this correct? it does equal params/bse
# but isn't the same as the AR example (which was wrong in the first place..)
print results.Tcontrast([0,1])  # are sd and t correct? vs
print results.Fcontrast(np.eye(2))


rhotrue = 0.99
beta = np.array([0.1, 2])
noiseratio = 1.
nsample = 20
x = np.arange(nsample)
X1 = add_constant(x)

noise = noiseratio * np.random.randn(nsample+1)
noise = noise[1:] + rhotrue*noise[:-1]
y = np.dot(X1,beta) + noise

mod1 = AR(y, X1, 1)
print mod1.results.params
print mod1.rho

for i in range(10):
    mod1.iterative_fit(1)
    print mod1.rho
    print mod1.results.params

print '\n iterative fitting of first model'
print 'with AR(0)', par0
parold = par0
mod0 = AR(Y, X, 1)
for i in range(10):
    print mod0.wdesign.sum()
    print mod0.calc_params.sum()
    mod0.iterative_fit(1)
    print mod0.rho
    parnew = mod0.results.params
    print parnew
    print parnew - parold
    parold = parnew



#example with no regressor, results incorrect, params=0 ?
model3 = AR(Y, rho=2)
for i in range(10):
    results = model3.fit()
    print "AR coefficients:", model3.rho, results.params
    rho, sigma = yule_walker(results.resid, order = model3.order)
    model3 = AR(Y, rho=rho)

xx = X.copy()
rhoyw, sigmayw = yule_walker(xx[:,0], order = 2)
print rhoyw, sigmayw
print (xx == X).all()  # test for unchanged array (fixed)



yy = Y.copy()
rhoyw, sigmayw = yule_walker(yy, order = 2)
print rhoyw, sigmayw
print (yy == Y).all()  # test for unchanged array (fixed)

Ydemeaned = Y - Y.mean()
model4 = AR(Ydemeaned, rho=2)
for i in range(10):
    results = model4.fit()
    print "AR coefficients:", model3.rho, results.params
    rho, sigma = yule_walker(results.resid, order = model4.order)
    model4 = AR(Ydemeaned, rho=rho)
