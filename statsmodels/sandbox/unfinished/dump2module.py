
if __name__ == '__main__':
    data = np.load(r"E:\Josef\eclipsegworkspace\statsmodels-josef-experimental-030\dist\statsmodels-0.3.0dev_with_Winhelp_a2\statsmodels-0.3.0dev\scikits\statsmodels\tsa\vector_ar\tests\results\vars_results.npz")
    res_var =  HoldIt('var_results')
    for d in data:
        setattr(res_var, d, data[d])
    np.set_printoptions(precision=120, linewidth=100)
    res_var.save(filename='testsave.py', header=True,
                  comment='VAR test data converted from vars_results.npz')

    import testsave

    for d in data:
        print(d)
        correct = np.all(data[d] == getattr(testsave.var_results, d))
        if not correct and not data[d].dtype == np.dtype('object'):
            correct = np.allclose(data[d], getattr(testsave.var_results, d),
                              rtol=1e-16, atol=1e-16)
            if not correct: print("inexact precision")
        if not correct:
            correlem =[np.all(data[d].item()[k] ==
                              getattr(testsave.var_results, d).item()[k])
                       for k in iterkeys(data[d].item())]
            if not correlem:
                print(d, "wrong")

    print(res_var.verify())

