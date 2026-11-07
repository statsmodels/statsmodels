# -*- coding: utf-8 -*-
"""Classes for transformations of variables or parameters


Created on Wed Jan  6 16:12:17 2021

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
# from statsmodels.genmod.families import links


class Transformation(object):

    def __call__(self, x, params):
        return self.trsansform(x, params)

    def transform(self, x, params):
        raise NotImplementedError

    def inverse(self, z, params):
        raise NotImplementedError

    def deriv(self, x, params):
        raise NotImplementedError

    def deriv_inverse(self, z, params):
        raise NotImplementedError

    def deriv2(self, x, params):
        raise NotImplementedError

    def deriv2_inverse(self, z, params):
        raise NotImplementedError


class Inverted(Transformation):
    """Inverted Transformation

    This creates a Transformation instance with reverted direction of
    transformation, i.e. the `transform` becomes the `inverse` transformation
    and vice versa.

    This does not create a new class. An instance of the original transform
    class is attached to the inverted instance and used for delegation.

    """

    def __init__(self, kls):
        self.original = kls()

        ss = self.original.direction.split(" ")
        self.direction = "%s to %s" % (ss[-1], ss[0])
        self.name = "Inverted " + kls.__name__

    def transform(self, x, params):
        return self.original.inverse(x, params)

    def inverse(self, z, params):
        return self.original.transform(z, params)

    def deriv(self, x, params):
        return self.original.deriv_inverse(x, params)

    def deriv_inverse(self, z, params):
        return self.original.deriv(z, params)

    def deriv2(self, x, params):
        return self.original.deriv2_inverse(x, params)

    def deriv2_inverse(self, z, params):
        return self.original.deriv2(z, params)


class SinhArcsinh(Transformation):
    """Sinh-arcsinh transformation

    Transformation is from R to R, changing skewness and kurtosis.

    `params` contains two parameters controlling skew and kurtosis

    The transform is

    z = sinh(d * arcsinh(x) - e)

    The inverse transform is

    x = sinh(1 / d * (arcsinh(z) + e))

    where ``d, e = params``

    """

    direction = "R to R"

    def transform(self, x, params):
        d, e = params
        z = np.sinh(d * np.arcsinh(x) - e)
        return z

    def inverse(self, z, params):
        d, e = params
        xrev = np.sinh(1 / d * (np.arcsinh(z) + e))
        return xrev

    def deriv(self, x, params):
        d, e = params
        d_dx = (d * np.cosh(e - d * np.arcsinh(x))) / np.sqrt(x**2 + 1)
        return d_dx

    def deriv_inverse(self, z, params):
        d, e = params
        d_dz = np.cosh((np.arcsinh(z) + e) / d) / (d * np.sqrt(z**2 + 1))
        return d_dz

    def deriv2(self, x, params):
        d, e = params
        d2_dx = ((d**2 * (-np.sqrt(x**2 + 1)) * np.sinh(e - d * np.arcsinh(x))
                 - d * x * np.cosh(e - d * np.arcsinh(x))) / (x**2 + 1)**(3/2))
        return d2_dx

    def deriv2_inverse(self, z, params):
        d, e = params
        d2_dz = ((np.sqrt(z**2 + 1) * np.sinh((np.arcsinh(z) + e)/d) -
                 d * z * np.cosh((np.arcsinh(z) + e)/d)) /
                 (d**2 * (z**2 + 1)**(3/2)))
        return d2_dz


class Sinh(Transformation):
    """Sinh Transformation, Johnson's SU transformation

    The two parameters are loc and scale transformation of base standard
    distributions.

    changing loc from zero introduces skewness

    Transformation has two `params` which are loc and scale of the base
    distribution

    The transformation is

    z = sinh((x - m)/s)

    The inverse transformation is

    x = m + s * arcsinh(z)

    where ``m, s = params``

    """
    # Note: we could use static methods, without self,
    # currently self is not used, but we might want to store something in an
    # instance

    direction = "R to R"

    def transform(self, x, params):
        m, s = params
        z = np.sinh((x - m)/s)
        return z

    def inverse(self, z, params):
        m, s = params
        xrev = m + s * np.arcsinh(z)
        return xrev

    def deriv(self, x, params):
        m, s = params
        t = (x - m) / s
        corr_fact = 1 / s
        d_dx = np.cosh(t) * corr_fact
        return d_dx

    def deriv_inverse(self, z, params):
        m, s = params  # noqa: F841
        d_dz = s / np.sqrt(z**2 + 1)
        return d_dz

    def deriv2(self, x, params):
        m, s = params
        t = (x - m) / s
        corr_fact = 1 / s**2
        d2_dx = np.sinh(t) * corr_fact
        return d2_dx

    def deriv2_inverse(self, z, params):
        m, s = params  # noqa: F841
        d2_dz = - s * z / (z**2 + 1)**(3/2)
        return d2_dz


class BirnbaumSaunders(Transformation):
    """Birnbaum-Saunders Transformation

    The transformation has two params
    to verify:
    a is scale of base distribution
    b is scale of transformed distribution

    The `loc` of the base distribution is fixed at zero, or more precisely,
    outside of the transformation.

    The inverse transformation is

    x =  1 / a * (sqrt(z / b) - sqrt(b / z))

    The transformation is

    z = b * (a / 2 * x + sqrt((a / 2 * x)**2 + 1))**2

    where ``a, b = params``
    """

    direction = "R to R+"

    def transform(self, x, params):
        a, b = params
        z = b * (a / 2 * x + np.sqrt((a / 2 * x)**2 + 1))**2
        return z

    def inverse(self, z, params):
        a, b = params
        xrev = 1 / a * (np.sqrt(z / b) - np.sqrt(b / z))
        return xrev

    def deriv(self, x, params):
        a, b = params
        t = a * x / 2
        corr_fact = (a / 2)
        d_dx = (2 * b * (np.sqrt(t**2 + 1) + t)**2
                ) / np.sqrt(t**2 + 1) * corr_fact
        return d_dx

    def deriv_inverse(self, z, params):
        a, b = params
        d_dz = (z + b) / (2 * a * np.sqrt(b) * z**(1.5))
        return d_dz

    def deriv2(self, x, params):
        a, b = params
        t = a * x / 2
        corr_fact = (a / 2)**2
        d2_dx = (2 * b * (np.sqrt(t**2 + 1) + t)**2 *
                 (2 * np.sqrt(t**2 + 1) - t)
                 ) / (t**2 + 1)**(3/2) * corr_fact
        return d2_dx

    def deriv2_inverse(self, z, params):
        a, b = params
        d2_dz = -(3 * b + z) / (4 * a * np.sqrt(b) * z**(5/2))
        return d2_dz


class _TMI1(Transformation):
    """T minus inverse transform `t - 1 / t`

    mapping from R to R+

    This transformation currently has no parameters, `params` is ignored.
    This is same transformation as TMI except for a different scaling
    coefficient.

    The transformation is

    z = 1/2 * (x - sqrt(x**2 + 4))

    The inverse transformation is

    x = t - 1 / t
    """
    # Note: we could use static methods, without self,
    # currently self is not used, but we might want to store something in an
    # instance

    direction = "R to R+"

    def transform(self, x, params=(None, None)):
        z = 1/2 * (x - np.sqrt(x**2 + 4))
        return z

    def inverse(self, z, params=(None, None)):
        t = z
        xrev = (t - 1 / t)
        return xrev

    def deriv(self, x, params=(None, None)):
        d_dx = 1/2 * (1 - x / np.sqrt(x**2 + 4))
        return d_dx

    def deriv_inverse(self, z, params=(None, None)):
        t = z
        d_dz = 1 / t**2 + 1
        return d_dz

    def deriv2(self, x, params=(None, None)):
        d2_dx = -2 / (x**2 + 4)**(3/2)
        return d2_dx

    def deriv2_inverse(self, z, params=(None, None)):
        t = z
        d2_dz = -2 / t**3
        return d2_dz


class TMI(Transformation):
    """inverse of `T minus inverse` transform `0.5 * (t - 1 / t)`,

    mapping from R to R+

    This transformation currently has no parameters, `params` is ignored.
    This transform `TMI` multiplies inverse transform by 1/2 compared to
    `TMI1`.

    Reference jones 2007

    The transformation is

    z = x - sqrt(x**2 + 1)

    The inverse transformation is

    x = 1/2 * (t - 1 / t)

    """

    direction = "R to R+"

    def transform(self, x, params):
        z = x + np.sqrt(x**2 + 1)
        return z

    def inverse(self, z, params):
        t = z
        xrev = 1/2 * (t - 1 / t)
        return xrev

    def deriv(self, x, params):
        d_dx = 1 + x / np.sqrt(x**2 + 1)
        return d_dx

    def deriv_inverse(self, z, params):
        t = z
        d_dz = 0.5 * (1 / t**2 + 1)
        return d_dz

    def deriv2(self, x, params):
        d2_dx = 1 / (x**2 + 1)**(3/2)
        return d2_dx

    def deriv2_inverse(self, z, params):
        t = z
        d2_dz = - 1 / t**3
        return d2_dz


class RtoInterval(Transformation):
    """transformation mapping from R to interval (0, 1)

    There are problems in the transformation for 0 in R and corresponding
    0.5 in unit interval.
    At zero the transformation is undefined and not well behaved,
    Taylor series expansion is well behaved, with finite values.

    inverse transform is well behaved in open interval (0, 1)

    This transformation currently has no parameters, `params` is ignored.
    This might need an additional loc-scale of base distribution specified by
    params.

    The transformation is

    z = (x - 1 + sqrt(x**2 + 1)) / (2 * x)

    The inverse transformation is

    x = 1/2 * (1 / (1 - t) - 1 / t)

    Reference Jones 2007
    """

    direction = "R to Interval"

    def transform(self, x, params=(None, None)):
        x = np.atleast_1d(x)  # needed for zero handling
        # z = (x - 1 + np.sqrt(x**2 + 1)) / (2 * x)
        # workaround, todo use lazywhere instead
        xnz = x.copy()
        xnz[x == 0] = 1e-10
        z = (xnz - 1 + np.sqrt(xnz**2 + 1)) / (2 * xnz)
        z[x == 0] = 0.5
        # Taylor series expansion around 0
        # maybe use this for a neighborhood of 0
        # z = 1 / 2 + x / 4 - x**3 / 16 + x**5 / 32
        return z

    def inverse(self, z, params=(None, None)):
        t = z
        xrev = 1/2 * (1 / (1 - t) - 1 / t)
        return xrev

    def deriv(self, x, params=(None, None)):
        x = np.atleast_1d(x)  # needed for zero handling
        d_dx = (1 - 1 / np.sqrt(x**2 + 1)) / (2 * x**2)
        d_dx[x == 0] = 0.25
        # Taylor series expansion around 0
        # maybe use this for a neighborhood or 0
        # d_dx = 1 / 4 - 3 * x**2 / 16 + 5 * x**4 / 32
        return d_dx

    def deriv_inverse(self, z, params=(None, None)):
        t = z
        d_dz = 0.5 * (1 / t**2 + 1 / (1 - t)**2)
        return d_dz

    def deriv2(self, x, params=(None, None)):
        x = np.atleast_1d(x)  # needed for zero handling
        d2_dx = ((3 - 2 * np.sqrt(x**2 + 1)) * x**2 -
                 2 * np.sqrt(x**2 + 1) + 2) / (2 * x**3 * (x**2 + 1)**(3/2))
        d2_dx[x == 0] = 0
        # Taylor series expansion around 0
        # maybe use this for a neighborhood or 0
        # d2_dx = - 3 * x / 8 + 5 * x**3 / 8 - 105 * x**5 / 128
        return d2_dx

    def deriv2_inverse(self, z, params=(None, None)):
        t = z
        d2_dz = 1 / (1 - t)**3 - 1 / t**3
        return d2_dz
