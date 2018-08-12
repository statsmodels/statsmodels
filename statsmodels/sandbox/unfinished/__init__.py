"""
CODE IN sandbox.unfinished IS NOT INTENDED TO BE USED BY ANYONE, FOR ANY
REASON, EVER.

This is unfinished, deprecated, or otherwise defunct code "rescued" from
other modules in the hopes that it may be made into tests or examples at
some point.
"""


# Taken from discrete.tests.test_constrained
def junk():
    # Singular Matrix in mod1a.fit()

    formula1 = 'deaths ~ smokes + C(agecat)'

    formula2 = 'deaths ~ C(agecat) + C(smokes) : C(agecat)'  # same as Stata default

    mod = Poisson.from_formula(formula2, data=data, exposure=data['pyears'].values)

    res0 = mod.fit()

    constraints = 'C(smokes)[T.1]:C(agecat)[3] = C(smokes)[T.1]:C(agecat)[4]'

    import patsy
    lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constraints)
    R, q = lc.coefs, lc.constants

    resc = mod.fit_constrained(R,q, fit_kwds={'method':'bfgs'})

    # example without offset
    formula1a = 'deaths ~ logpyears + smokes + C(agecat)'
    mod1a = Poisson.from_formula(formula1a, data=data)
    print(mod1a.exog.shape)

    res1a = mod1a.fit()
    lc_1a = patsy.DesignInfo(mod1a.exog_names).linear_constraint('C(agecat)[T.4] = C(agecat)[T.5]')
    resc1a = mod1a.fit_constrained(lc_1a.coefs, lc_1a.constants, fit_kwds={'method':'newton'})
    print(resc1a[0])
    print(resc1a[1])
