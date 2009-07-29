import numpy as np
from models.functions import add_constant
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
