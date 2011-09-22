'''examples to check summary, not converted to tests yet


'''

if __name__ == '__main__':
    
    from scikits.statsmodels.regression.tests.test_regression  import TestOLS

    #def mytest():
    aregression = TestOLS()
    TestOLS.setupClass()
    results = aregression.res1
    r_summary = str(results.summary())
    print r_summary
    olsres = results

    print '\n\n'

    r_summary = str(results.summary2())
    print r_summary
    print '\n\n'


    from scikits.statsmodels.discrete.tests.test_discrete  import TestProbitNewton

    aregression = TestProbitNewton()
    TestProbitNewton.setupClass()
    results = aregression.res1
    r_summary = str(results.summary())
    print r_summary

    from scikits.statsmodels.robust.tests.test_rlm  import TestHampel

    aregression = TestHampel()
    #TestHampel.setupClass()
    results = aregression.res1
    r_summary = str(results.summary())
    print r_summary

    print '\n\n'

    from scikits.statsmodels.genmod.tests.test_glm  import TestGlmBinomial

    aregression = TestGlmBinomial()
    #TestGlmBinomial.setupClass()
    results = aregression.res1
    r_summary = str(results.summary2())
    print r_summary

    #print results.summary2(return_fmt='latex')
    #print results.summary2(return_fmt='csv')
    
    smry = olsres.summary2()
    print smry.as_csv()
    

