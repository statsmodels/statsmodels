import numpy as np

class RobustNorm:
    def __call__(self, z):
        return self.rho(z)

class LeastSquares(RobustNorm):

    """
    Least squares rho for M estimation.

    DC Montgomery, EA Peck. \'Introduction to Linear Regression Analysis\',
    John Wiley and Sons, Inc., New York, 2001.

    """

    def rho(self, z):
        return np.power(z, 2) * 0.5

    def psi(self, z):
        return np.asarray(z)

    def weights(self, z):
        return np.ones(z.shape, np.float64)

class HuberT(RobustNorm):

    """
    Huber\'s T for M estimation.

    DC Montgomery, EA Peck. \'Introduction to Linear Regression Analysis\',
    John Wiley and Sons, Inc., New York, 2001.

    R Venables, B Ripley. \'Modern Applied Statistics in S\'
    Springer, New York, 2002.

    """

    t = 1.345

    def subset(self, z):
        z = np.asarray(z)
        return np.less_equal(np.fabs(z), HuberT.t)

    def rho(self, z):
        z = np.asarray(z)
        test = self.subset(z)
        return (test * 0.5 * np.power(z, 2) +
                (1 - test) * (np.fabs(z) * HuberT.t - 0.5 * HuberT.t**2))

    def psi(self, z):
        z = np.asarray(z)
        test = self.subset(z)
        return test * z + (1 - test) * HuberT.t * np.sign(z)

    def weights(self, z):
        z = np.asarray(z)
        test = self.subset(z)
        return test + (1 - test) * HuberT.t / np.fabs(z)

class RamsayE(RobustNorm):

    """
    Ramsay\'s Ea for M estimation.

    DC Montgomery, EA Peck. \'Introduction to Linear Regression Analysis\',
    John Wiley and Sons, Inc., New York, 2001.

    """
    a = 0.3

    def rho(self, z):
        z = np.asarray(z)
        return (1 - np.exp(-RamsayE.a * np.fabs(z)) *
                (1 + RamsayE.a * np.fabs(z))) / RamsayE.a**2

    def psi(self, z):
        z = np.asarray(z)
        return z * np.exp(-RamsayE.a * np.fabs(z))

    def weights(self, z):
        z = np.asarray(z)
        return np.exp(-RamsayE.a * np.fabs(z))

class AndrewWave(RobustNorm):

    """
    Andrew\'s wave for M estimation.

    DC Montgomery, EA Peck. \'Introduction to Linear Regression Analysis\',
    John Wiley and Sons, Inc., New York, 2001.

    """
    a = 1.339

    def subset(self, z):
        z = np.asarray(z)
        return np.less_equal(np.fabs(z), RamsayE.a * np.pi)

    def rho(self, z):
        a = AndrewWave.a
        z = np.asarray(z)
        test = self.subset(z)
        return (test * a * (1 - np.cos(z / a)) +
                (1 - test) * 2 * a)

    def psi(self, z):
        a = AndrewWave.a
        z = np.asarray(z)
        test = self.subset(z)
        return test * np.sin(z / a)

    def weights(self, z):
        a = AndrewWave.a
        z = np.asarray(z)
        test = self.subset(z)
        return test * np.sin(z / a) / (z / a)

class TrimmedMean(RobustNorm):
    """

    Trimmed mean function for M-estimation.

    R Venables, B Ripley. \'Modern Applied Statistics in S\'
    Springer, New York, 2002.
    """

    c = 2

    def subset(self, z):
        z = np.asarray(z)
        return np.less_equal(np.fabs(z), TrimmedMean.c)

    def rho(self, z):
        z = np.asarray(z)
        test = self.subset(z)
        return test * np.power(z, 2) * 0.5

    def psi(self, z):
        z = np.asarray(z)
        test = self.subset(z)
        return test * z

    def weights(self, z):
        z = np.asarray(z)
        test = self.subset(z)
        return test

class Hampel(RobustNorm):
    """

    Hampel function for M-estimation.

    R Venables, B Ripley. \'Modern Applied Statistics in S\'
    Springer, New York, 2002.
    """

    a = 2
    b = 4
    c = 8

    def subset(self, z):
        z = np.fabs(np.asarray(z))
        t1 = np.less_equal(z, Hampel.a)
        t2 = np.less_equal(z, Hampel.b) * np.greater(z, Hampel.a)
        t3 = np.less_equal(z, Hampel.c) * np.greater(z, Hampel.b)
        return t1, t2, t3

    def psi(self, z):
        z = np.asarray(z)
        a = Hampel.a; b = Hampel.b; c = Hampel.c
        t1, t2, t3 = self.subset(z)
        s = np.sign(z); z = np.fabs(z)
        v = s * (t1 * z +
                 t2 * a +
                 t3 * a * (c - z) / (c - b))
        return v

    def rho(self, z):
        z = np.fabs(z)
        a = Hampel.a; b = Hampel.b; c = Hampel.c
        t1, t2, t3 = self.subset(z)
        v = (t1 * z**2 * 0.5 +
             t2 * (a * z - a**2 * 0.5) +
             t3 * (a * (c * z - z**2 * 0.5) / (c - b) - 7 * a**2 / 2.) +
             (1 - t1 + t2 + t3) * a * (b + c - a))
        return v

    def weights(self, z):
        z = np.asarray(z)
        test = np.not_equal(z, 0)
        return self.psi(z) * test / z + (1 - test)

class TukeyBiweight(RobustNorm):
    """

    Tukey\'s biweight function for M-estimation.

    R Venables, B Ripley. \'Modern Applied Statistics in S\'
    Springer, New York, 2002.
    """


    R = 4.685

    def subset(self, z):
        z = np.fabs(np.asarray(z))
        return np.less_equal(z, self.R)

    def psi(self, z):
        z = np.asarray(z)
        subset = self.subset(z)
        return z * (1 - (z / self.R)**2)**2 * subset

    def rho(self, z):
        subset = self.subset(z)
        return -(1 - (z / self.R)**2)**3 * subset * self.R**2 / 6

    def weights(self, z):
        subset = self.subset(z)
        return (1 - (z / self.R)**2)**2 * subset
