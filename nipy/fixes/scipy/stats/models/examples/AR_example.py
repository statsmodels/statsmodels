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
results.params
results.t() # is this correct? it does equal params/bse
# but isn't the same as the AR example (which was wrong in the first place..)
print results.Tcontrast([0,1])  # are sd and t correct? vs
print results.Fcontrast(np.eye(2))


rhotrue = 0.7
beta = np.array([1, 2])
noiseratio = 0.1
nsample = 100
x = np.arange(nsample)
X2 = add_constant(x)

noise = noiseratio * np.random.randn(nsample+1)
noise = noise[1:] + rhotrue*noise[:-1]
y = np.dot(X2,beta) + noise

mod1 = AR(y, X2, 1)
print mod1.results.params
print mod1.rho
