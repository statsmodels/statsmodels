
## Contrasts Overview
from __future__ import print_function
import numpy as np
import statsmodels.api as sm


# This document is based heavily on this excellent resource from UCLA http://www.ats.ucla.edu/stat/r/library/contrast_coding.htm

# A categorical variable of K categories, or levels, usually enters a regression as a sequence of K-1 dummy variables. This amounts to a linear hypothesis on the level means. That is, each test statistic for these variables amounts to testing whether the mean for that level is statistically significantly different from the mean of the base category. This dummy coding is called Treatment coding in R parlance, and we will follow this convention. There are, however, different coding methods that amount to different sets of linear hypotheses.
#
# In fact, the dummy coding is not technically a contrast coding. This is because the dummy variables add to one and are not functionally independent of the model's intercept. On the other hand, a set of *contrasts* for a categorical variable with `k` levels is a set of `k-1` functionally independent linear combinations of the factor level means that are also independent of the sum of the dummy variables. The dummy coding isn't wrong *per se*. It captures all of the coefficients, but it complicates matters when the model assumes independence of the coefficients such as in ANOVA. Linear regression models do not assume independence of the coefficients and thus dummy coding is often the only coding that is taught in this context.
#
# To have a look at the contrast matrices in Patsy, we will use data from UCLA ATS. First let's load the data.

##### Example Data

import pandas as pd
url = 'http://www.ats.ucla.edu/stat/data/hsb2.csv'
hsb2 = pd.read_table(url, delimiter=",")


hsb2.head(10)


# It will be instructive to look at the mean of the dependent variable, write, for each level of race ((1 = Hispanic, 2 = Asian, 3 = African American and 4 = Caucasian)).

hsb2.groupby('race')['write'].mean()


##### Treatment (Dummy) Coding

# Dummy coding is likely the most well known coding scheme. It compares each level of the categorical variable to a base reference level. The base reference level is the value of the intercept. It is the default contrast in Patsy for unordered categorical factors. The Treatment contrast matrix for race would be

from patsy.contrasts import Treatment
levels = [1,2,3,4]
contrast = Treatment(reference=0).code_without_intercept(levels)
print(contrast.matrix)


# Here we used `reference=0`, which implies that the first level, Hispanic, is the reference category against which the other level effects are measured. As mentioned above, the columns do not sum to zero and are thus not independent of the intercept. To be explicit, let's look at how this would encode the `race` variable.

hsb2.race.head(10)


print(contrast.matrix[hsb2.race-1, :][:20])


sm.categorical(hsb2.race.values)


# This is a bit of a trick, as the `race` category conveniently maps to zero-based indices. If it does not, this conversion happens under the hood, so this won't work in general but nonetheless is a useful exercise to fix ideas. The below illustrates the output using the three contrasts above

from statsmodels.formula.api import ols
mod = ols("write ~ C(race, Treatment)", data=hsb2)
res = mod.fit()
print(res.summary())


# We explicitly gave the contrast for race; however, since Treatment is the default, we could have omitted this.

#### Simple Coding

# Like Treatment Coding, Simple Coding compares each level to a fixed reference level. However, with simple coding, the intercept is the grand mean of all the levels of the factors. Patsy doesn't have the Simple contrast included, but you can easily define your own contrasts. To do so, write a class that contains a code_with_intercept and a code_without_intercept method that returns a patsy.contrast.ContrastMatrix instance

from patsy.contrasts import ContrastMatrix

def _name_levels(prefix, levels):
    return ["[%s%s]" % (prefix, level) for level in levels]

class Simple(object):
    def _simple_contrast(self, levels):
        nlevels = len(levels)
        contr = -1./nlevels * np.ones((nlevels, nlevels-1))
        contr[1:][np.diag_indices(nlevels-1)] = (nlevels-1.)/nlevels
        return contr

    def code_with_intercept(self, levels):
        contrast = np.column_stack((np.ones(len(levels)),
                                    self._simple_contrast(levels)))
        return ContrastMatrix(contrast, _name_levels("Simp.", levels))

    def code_without_intercept(self, levels):
        contrast = self._simple_contrast(levels)
        return ContrastMatrix(contrast, _name_levels("Simp.", levels[:-1]))


hsb2.groupby('race')['write'].mean().mean()


contrast = Simple().code_without_intercept(levels)
print(contrast.matrix)


mod = ols("write ~ C(race, Simple)", data=hsb2)
res = mod.fit()
print(res.summary())


#### Sum (Deviation) Coding

# Sum coding compares the mean of the dependent variable for a given level to the overall mean of the dependent variable over all the levels. That is, it uses contrasts between each of the first k-1 levels and level k In this example, level 1 is compared to all the others, level 2 to all the others, and level 3 to all the others.

from patsy.contrasts import Sum
contrast = Sum().code_without_intercept(levels)
print(contrast.matrix)


mod = ols("write ~ C(race, Sum)", data=hsb2)
res = mod.fit()
print(res.summary())


# This corresponds to a parameterization that forces all the coefficients to sum to zero. Notice that the intercept here is the grand mean where the grand mean is the mean of means of the dependent variable by each level.

hsb2.groupby('race')['write'].mean().mean()


#### Backward Difference Coding

# In backward difference coding, the mean of the dependent variable for a level is compared with the mean of the dependent variable for the prior level. This type of coding may be useful for a nominal or an ordinal variable.

from patsy.contrasts import Diff
contrast = Diff().code_without_intercept(levels)
print(contrast.matrix)


mod = ols("write ~ C(race, Diff)", data=hsb2)
res = mod.fit()
print(res.summary())


# For example, here the coefficient on level 1 is the mean of `write` at level 2 compared with the mean at level 1. Ie.,

res.params["C(race, Diff)[D.1]"]
hsb2.groupby('race').mean()["write"][2] -      hsb2.groupby('race').mean()["write"][1]


#### Helmert Coding

# Our version of Helmert coding is sometimes referred to as Reverse Helmert Coding. The mean of the dependent variable for a level is compared to the mean of the dependent variable over all previous levels. Hence, the name 'reverse' being sometimes applied to differentiate from forward Helmert coding. This comparison does not make much sense for a nominal variable such as race, but we would use the Helmert contrast like so:

from patsy.contrasts import Helmert
contrast = Helmert().code_without_intercept(levels)
print(contrast.matrix)


mod = ols("write ~ C(race, Helmert)", data=hsb2)
res = mod.fit()
print(res.summary())


# To illustrate, the comparison on level 4 is the mean of the dependent variable at the previous three levels taken from the mean at level 4

grouped = hsb2.groupby('race')
grouped.mean()["write"][4] - grouped.mean()["write"][:3].mean()


# As you can see, these are only equal up to a constant. Other versions of the Helmert contrast give the actual difference in means. Regardless, the hypothesis tests are the same.

k = 4
1./k * (grouped.mean()["write"][k] - grouped.mean()["write"][:k-1].mean())
k = 3
1./k * (grouped.mean()["write"][k] - grouped.mean()["write"][:k-1].mean())


#### Orthogonal Polynomial Coding

# The coefficients taken on by polynomial coding for `k=4` levels are the linear, quadratic, and cubic trends in the categorical variable. The categorical variable here is assumed to be represented by an underlying, equally spaced numeric variable. Therefore, this type of encoding is used only for ordered categorical variables with equal spacing. In general, the polynomial contrast produces polynomials of order `k-1`. Since `race` is not an ordered factor variable let's use `read` as an example. First we need to create an ordered categorical from `read`.

hsb2['readcat'] = pd.cut(hsb2.read, bins=3)
hsb2.groupby('readcat').mean()['write']


from patsy.contrasts import Poly
levels = hsb2.readcat.unique().tolist()
contrast = Poly().code_without_intercept(levels)
print(contrast.matrix)


mod = ols("write ~ C(readcat, Poly)", data=hsb2)
res = mod.fit()
print(res.summary())


# As you can see, readcat has a significant linear effect on the dependent variable `write` but not a significant quadratic or cubic effect.
