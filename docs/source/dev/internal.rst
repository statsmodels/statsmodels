.. _model:



Internal Classes
================

Introduction
------------

The following summarizes classes and functions that are not intended to be
directly used, but of interest only for internal use or for a developer who
wants to extend on existing model classes.


Module Reference
----------------

Model and Results Classes
^^^^^^^^^^^^^^^^^^^^^^^^^

These are the base classes for both the estimation models and the results.
They are not directly useful, but layout the structure of the subclasses and
define some common methods.

.. currentmodule:: statsmodels.base.model

.. autosummary::
   :toctree: generated/

   Model
   LikelihoodModel
   GenericLikelihoodModel
   Results
   LikelihoodModelResults
   ResultMixin
   GenericLikelihoodModelResults

.. inheritance-diagram:: statsmodels.base.model statsmodels.discrete.discrete_model statsmodels.regression.linear_model statsmodels.miscmodels.count
   :parts: 3
