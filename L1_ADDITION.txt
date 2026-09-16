What the l1 addition is
=======================
A slight modification that allows l1 regularized LikelihoodModel.

Regularization is handled by a fit_regularized method.

Main Files
==========

l1_demo/demo.py
    $ python demo.py --get_l1_slsqp_results logit
    does a quick demo of the regularization using logistic regression.

l1_demo/sklearn_compare.py
    $ python sklearn_compare.py
    Plots a comparison of regularization paths.  Modify the source to use
    different datasets.

statsmodels/base/l1_cvxopt.py
    fit_l1_cvxopt_cp()
        Fit likelihood model using l1 regularization.  Use the CVXOPT package.
    Lots of small functions supporting fit_l1_cvxopt_cp

statsmodels/base/l1_slsqp.py
    fit_l1_slsqp()
        Fit likelihood model using l1 regularization.  Use scipy.optimize
    Lots of small functions supporting fit_l1_slsqp

statsmodels/base/l1_solvers_common.py
    Common methods used by l1 solvers

statsmodels/base/model.py
    Likelihoodmodel.fit()
        3 lines modified to allow for importing and calling of l1 fitting functions

statsmodels/discrete/discrete_model.py
    L1MultinomialResults class
        Child of MultinomialResults
    MultinomialModel.fit()
        3 lines re-directing l1 fit results to the L1MultinomialResults class
