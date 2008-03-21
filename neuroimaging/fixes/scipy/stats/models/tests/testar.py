from exampledata import x, y
import neuroimaging.fixes.scipy.stats_models as SSM

for i in range(1,4):
    model = SSM.regression.ARModel(x, i)
    for i in range(20):
        results = model.fit(y)
        rho, sigma = model.yule_walker(y - results.predict)
        model = SSM.regression.ARModel(model.design, rho)
    print "AR coefficients:", model.rho

