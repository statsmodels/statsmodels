What the l1 addition is
=======================
A slight modification that allows l1 regularized multinomial logistic regression.  The functionality should be compatible with the existing base.LikelihoodModel.

Main Files
==========

statsmodels/discrete/l1.py
    _fit_l1() : 
        Similar call structure to e.g. _fit_mle_newton()
    Lots of small functions supporting _fit_l1
    modified_bic(), modified_aic(), modified_df_model :
        Modifications because a number of parameters will be set to zero by the l1 regularization

statsmodels/base/model.py
    Likelihoodmodel.fit() :
        3 lines modified to allow for importing and calling of l1.py:_fit_l1()
