Patsy: Contrast Coding Systems for categorical variables
===========================================================

.. note:: This document is based heavily on `this excellent resource from UCLA <http://www.ats.ucla.edu/stat/r/library/contrast_coding.htm>__`.

A categorical variable of K categories, or levels, usually enters a regression as a sequence of K-1 dummy variables. This amounts to a linear hypothesis on the level means. That is, each test statistic for these variables amounts to testing whether the mean for that level is statistically significantly different from the mean of the base category. This dummy coding is called Treatment coding in R parlance, and we will follow this convention. There are, however, different coding methods that amount to different sets of linear hypotheses. 

In fact, the dummy coding is not technically a contrast coding. This is because the dummy variables add to one and are not functionally independent of the model's intercept. On the other hand, a set of *contrasts* for a categorical variable with `k` levels is a set of `k-1` functionally independent linear combinations of the factor level means that are also independent of the sum of the dummy variables. The dummy coding isn't wrong *per se*. It captures all of the coefficients, but it complicates matters when the model assumes independence of the coefficients such as in ANOVA. Linear regression models do not assume independence of the coefficients and thus dummy coding is often the only coding that is taught in this context.

To have a look at the contrast matrices in Patsy, we will use data from UCLA ATS. First let's load the data.

Example Data
------------

.. ipython:: python

   import pandas # requires current master for URL
   url = 'http://www.ats.ucla.edu/stat/R/notes/hsb2_nolabel.csv'
   try:
       hsb2 = pandas.read_table(url, delimiter=",")
   except:
       from urllib2 import urlopen
       hsb2 = pandas.read_table(urlopen(url), delimiter=",")

It will be instructive to look at the mean of the dependent variable, write, for each level of race ((1 = Hispanic, 2 = Asian, 3 = African American and 4 = Caucasian)).

.. ipython::

   hsb2.groupby('race')['write'].mean()

Treatment (Dummy) Coding
------------------------

Dummy coding is likely the most well known coding scheme. It compares each level of the categorical variable to a base reference level. The base reference level is the value of the intercept. It is the default contrast in Patsy for unordered categorical factors. The Treatment contrast matrix for race would be

.. ipython:: python

   from patsy.contrasts import Treatment
   levels = [1,2,3,4]
   contrast = Treatment(base=0).code_without_intercept(levels)
   print contrast.matrix

Here we used `base=0`, which implies that the first level, Hispanic, is the reference category against which the other level effects are measured. As mentioned above, the columns do not sum to zero and are thus not independent of the intercept. To be explicit, let's look at how this would encode the `race` variable.

.. ipython:: python

   contrast.matrix[hsb2.race-1, :]

This is a bit of a trick, as the `race` category conveniently maps to zero-based indices. If it does not, this conversion happens under the hood, so this won't work in general but nonetheless is a useful exercise to fix ideas. The below illustrates the output using the three contrasts above

.. ipython:: python

   from statsmodels.formula.api import ols
   mod = ols("write ~ C(race, Treatment)", df=hsb2)
   res = mod.fit()
   print res.summary()

We explicitly gave the contrast for race; however, since Treatment is the default, we could have omitted this.

Simple Coding
-------------

Like Treatment Coding, Simple Coding compares each level to a fixed reference level. However, with simple coding, the intercept is the grand mean of all the levels of the factors.

.. ipython:: python

   from patsy.contrasts import Simple
   contrast = Simple().code_without_intercept(levels)
   print contrast.matrix

   mod = ols("write ~ C(race, Simple)", df=hsb2)
   res = mod.fit()
   print res.summary()

Sum (Deviation) Coding
----------------------

Sum coding compares the mean of the dependent variable for a given level to the overall mean of the dependent variable over all the levels. That is, it uses contrasts between each of the first k-1 levels and level k In this example, level 1 is compared to all the others, level 2 to all the others, and level 3 to all the others.

.. ipython:: python

   from patsy.contrasts import Sum
   contrast = Sum().code_without_intercept(levels)
   print contrast.matrix

   mod = ols("write ~ C(race, Sum)", df=hsb2)
   res = mod.fit()
   print res.summary()

This correspons to a parameterization that forces all the coefficients to sum to zero. Notice that the intercept here is the grand mean where the grand mean is the mean of means of the dependent variable by each level.

.. ipython:: python

   hsb2.groupby('race')['write'].mean().mean()

Backward Difference Coding
--------------------------

In backward difference coding, the mean of the dependent variable for a level is compared with the mean of the dependent variable for the prior level. This type of coding may be useful for a nominal or an ordinal variable.

.. ipython:: python

   from patsy.contrasts import BDiff
   contrast = BDiff().code_without_intercept(levels)
   print contrast.matrix

   mod = ols("write ~ C(race, BDiff)", df=hsb2)
   res = mod.fit()
   print res.summary()

For example, here the coefficient on level 1 is the mean of `write` at level 2 compared with the mean at level 1. Ie.,

.. ipython:: python

   res.params["C(race, BDiff)[D.1]"]
   hsb2.groupby('race').mean()["write"][2] - \
        hsb2.groupby('race').mean()["write"][1]

Helmert Coding
--------------

Our version of Helmert coding is sometimes referred to as Reverse Helmert Coding. The mean of the dependent variable for a level is compared to the mean of the dependent variable over all previous levels. Hence, the name 'reverse' being sometimes applied to differentiate from forward Helmert coding. This comparison does not make much sense for a nominal variable such as race, but we would use the Helmert contrast like so:

.. ipython:: python

   from patsy.contrasts import Helmert
   contrast = Helmert().code_without_intercept(levels)
   print contrast.matrix

   mod = ols("write ~ C(race, Helmert)", df=hsb2)
   res = mod.fit()
   print res.summary()

To illustrate, the comparison on level 4 is the mean of the dependent variable at the previous three levels taken from the mean at level 4

.. ipython:: python

   grouped = hsb2.groupby('race')
   grouped.mean()["write"][4] - grouped.mean()["write"][:3].mean()

As you can see, these are only equal up to a constant. Other versions of the Helmert contrast give the actual difference in means. Regardless, the hypothesis tests are the same.

.. ipython:: python

   k = 4
   1./k * (grouped.mean()["write"][k] - grouped.mean()["write"][:k-1].mean())
   k = 3
   1./k * (grouped.mean()["write"][k] - grouped.mean()["write"][:k-1].mean())

   
Orthogonal Polynomial Coding
----------------------------

The coefficients taken on by polynomial coding for `k=4` levels are the linear, quadratic, and cubic trends in the categorical variable. The categorical variable here is assumed to be represented by an underlying, equally spaced numeric variable. Therefore, this type of encoding is used only for ordered categorical variables with equal spacing. In general, the polynomial contrast produces polynomials of order `k-1`. Since `race` is not an ordered factor variable let's use `read` as an example. First we need to create an ordered categorical from `read`.

.. ipython:: python

   _, bins = np.histogram(hsb2.read, 3)
   try: # requires numpy master
       readcat = np.digitize(hsb2.read, bins, True)
   except:
       readcat = np.digitize(hsb2.read, bins)
   hsb2['readcat'] = readcat
   hsb2.groupby('readcat').mean()['write']

.. ipython:: python

   from patsy.contrasts import Poly
   levels = hsb2.readcat.unique().tolist()
   contrast = Poly().code_without_intercept(levels)
   print contrast.matrix

   mod = ols("write ~ C(readcat, Poly)", df=hsb2)
   res = mod.fit()
   print res.summary()

As you can see, readcat has a significant linear effect on the dependent variable `write` but not a significant quadratic or cubic effect.

User-Defined Coding
-------------------

Right now, if you want to use your own coding, you must do so by writing a coding class that contains a code_with_intercept and a code_without_intercept method that return a patsy.contrast.ContrastMatrix instance. To use this custom coding you would do


.. code:: python

   from patsy.state import builtin_stateful_transforms
   builtin_stateful_transforms["MyContrast"] = MyContrast

   mod = ols("write ~ C(race, MyContrast)", df=hsb2)
   res = mod.fit()
