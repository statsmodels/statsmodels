Source: https://github.com/statsmodels/statsmodels/issues/8506
Original issue author: mfissore, 2022-11-07.
The CSV retains the issue's 60 response observations and nested factor labels.
The observation-number column, omitted here, is 0..4 repeated twelve times.

Reference: https://github.com/statsmodels/statsmodels/issues/8506#issuecomment-1307746481
Maintainer josef-pkt recoded B to reuse labels within A, giving full column rank:
    A: df=2, sum_sq=441.2333333333333
    A:B: df=9, sum_sq=2656.9
    Residual: df=48, sum_sq=18388.8
These are fixed-effects tests with residual mean square 383.1.

The expectations can also be computed without a model fit. Each A level has
20 observations and each nested B level has 5 observations. Let m be the grand
mean, m_a the mean within A, and m_ab the mean in each nested cell. Then:
    SS_A = 20 * sum_a (m_a - m)**2
    SS_AB = 5 * sum_ab (m_ab - m_a)**2
    SS_residual = sum_i (y_i - m_ab[i])**2
The test suite additionally uses these group-mean formulas when levels and
observations are removed to make the original data unbalanced.
