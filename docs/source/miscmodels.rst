

.. currentmodule:: statsmodels.miscmodels


.. _miscmodels:


Other Models :mod:`miscmodels`
==============================

:mod:`scikits.statmodels.miscmodels` contains model classes and that do not yet fit into
any other category, or are basic implementations that are not yet polished and will most
likely still change. Some of these models were written as examples for the generic
maximum likelihood framework, and there will be others that might be based on general
method of moments.

The models in this category have been checked for basic cases, but might be more exposed
to numerical problems than the complete implementation. For example, count.Poisson has
been added using only the generic maximum likelihood framework, the standard errors
are based on the numerical evaluation of the Hessian, while discretemod.Poisson uses
analytical Gradients and Hessian and will be more precise, especially in cases when there
is strong multicollinearity.
On the other hand, by subclassing GenericLikelihoodModel, it is easy to add new models,
another example can be seen in the zero inflated Poisson model, miscmodels.count.


Count Models :mod:`count`
--------------------------

.. currentmodule:: statsmodels.miscmodels.count

.. autosummary::
   :toctree: generated/

   PoissonGMLE
   PoissonOffsetGMLE
   PoissonZiGMLE

Linear Model with t-distributed errors
--------------------------------------

This is a class that shows that a new model can be defined by only specifying the
method for the loglikelihood. All result statistics are inherited from the generic
likelihood model and result classes. The results have been checked against R for a
simple case.

.. currentmodule:: statsmodels.miscmodels.tmodel

.. autosummary::
   :toctree: generated/

   TLinearModel




