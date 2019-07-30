.. currentmodule:: statsmodels.gam.api

.. _gam:

Generalized Additive Models (GAM)
=================================

Generalized Additive Models allow for penalized estimation of smooth terms
in generalized linear models.

See `Module Reference`_ for commands and arguments.

Examples
--------

The following illustrates a Gaussian and a Poisson regression where
categorical variables are treated as linear terms and the effect of
two explanatory variables is captured by penalized B-splines.
The data is from the automobile dataset
https://archive.ics.uci.edu/ml/datasets/automobile
We can load a dataframe with selected columns from the unit test module.

.. ipython:: python

    import statsmodels.api as sm
    from statsmodels.gam.api import GLMGam, BSplines

    # import data
    from statsmodels.gam.tests.test_penalized import df_autos

    # create spline basis for weight and hp
    x_spline = df_autos[['weight', 'hp']]
    bs = BSplines(x_spline, df=[12, 10], degree=[3, 3])

    # penalization weight
    alpha = np.array([21833888.8, 6460.38479])

    gam_bs = GLMGam.from_formula('city_mpg ~ fuel + drive', data=df_autos,
                                 smoother=bs, alpha=alpha)
    res_bs = gam_bs.fit()
    print(res_bs.summary())

    # plot smooth components
    res_bs.plot_partial(0, cpr=True)
    res_bs.plot_partial(1, cpr=True)

    alpha = np.array([8283989284.5829611, 14628207.58927821])
    gam_bs = GLMGam.from_formula('city_mpg ~ fuel + drive', data=df_autos,
                                 smoother=bs, alpha=alpha,
                                 family=sm.families.Poisson())
    res_bs = gam_bs.fit()
    print(res_bs.summary())

    # Optimal penalization weights alpha can be obtaine through generalized
    # cross-validation or k-fold cross-validation.
    # The alpha above are from the unit tests against the R mgcv package.

    gam_bs.select_penweight()[0]
    gam_bs.select_penweight_kfold()[0]


References
^^^^^^^^^^

* Hastie, Trevor, and Robert Tibshirani. 1986. Generalized Additive Models. Statistical Science 1 (3): 297-310.
* Wood, Simon N. 2006. Generalized Additive Models: An Introduction with R. Texts in Statistical Science. Boca Raton, FL: Chapman & Hall/CRC.
* Wood, Simon N. 2017. Generalized Additive Models: An Introduction with R. Second edition. Chapman & Hall/CRC Texts in Statistical Science. Boca Raton: CRC Press/Taylor & Francis Group.


Module Reference
----------------

.. module:: statsmodels.gam.generalized_additive_model
   :synopsis: Generalized Additive Models
.. currentmodule:: statsmodels.gam.generalized_additive_model

Model Class
^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   GLMGam
   LogitGam

Results Classes
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   GLMGamResults

Smooth Basis Functions
^^^^^^^^^^^^^^^^^^^^^^

.. module:: statsmodels.gam.smooth_basis
   :synopsis: Classes for Spline and other Smooth Basis Function

.. currentmodule:: statsmodels.gam.smooth_basis

Currently there is verified support for two spline bases

.. autosummary::
   :toctree: generated/

   BSplines
   CyclicCubicSplines

`statsmodels.gam.smooth_basis` includes additional splines and a (global)
polynomial smoother basis but those have not been verified yet.



Families and Link Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The distribution families in `GLMGam` are the same as for GLM and so are
the corresponding link functions.
Current unit tests only cover Gaussian and Poisson, and GLMGam might not
work for all options that are available in GLM.
