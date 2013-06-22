# test_lowess_r_output.R
#
# Generate outputs for unit tests
# for lowess function in cylowess.pyx
#
# May 2012
#

# test_simple
x_simple = 0:19
# Standard Normal noise
noise_simple = c(-0.76741118, -0.30754369,
                 0.39950921, -0.46352422, -1.67081778,
                 0.6595567 ,  0.66367639, -2.04388585,
                 0.8123281 ,  1.45977518,
                 1.21428038,  1.29296866,  0.78028477,
                 -0.2402853 , -0.21721302,
                 0.24549405,  0.25987014, -0.90709034,
                 -1.45688216, -0.31780505)

y_simple = x_simple + noise_simple

out_simple = lowess(x_simple, y_simple, delta = 0, iter = 3)


# test_iter
x_iter = 0:19
# Cauchy noise
noise_iter = c(1.86299605, -0.10816866,  1.87761229,
               -3.63442237,  0.30249022,
                1.03560416,  0.21163349,  1.14167809,
               -0.00368175, -2.08808987,
               0.13065417, -1.8052207 ,  0.60404596,
               -2.30908204,  1.7081412 ,
               -0.54633243, -0.93107948,  1.79023999,
               1.05822445, -1.04530564)

y_iter = x_iter + noise_iter

out_iter_0 = lowess(x_iter, y_iter, delta = 0, iter = 0)
out_iter_3 = lowess(x_iter, y_iter, delta = 0, iter = 3)


# test_frac
x_frac = seq(-2*pi, 2*pi, length = 30)

# normal noise
noise_frac  = c(1.62379338, -1.11849371,  1.60085673,
                0.41996348,  0.70896754,
                0.19271408,  0.04972776, -0.22411356,
                0.18154882, -0.63651971,
                0.64942414, -2.26509826,  0.80018964,
                0.89826857, -0.09136105,
                0.80482898,  1.54504686, -1.23734643,
                -1.16572754,  0.28027691,
                -0.85191583,  0.20417445,  0.61034806,
                0.68297375,  1.45707167,
                0.45157072, -1.13669622, -0.08552254,
               -0.28368514, -0.17326155)

y_frac = sin(x_frac) + noise_frac

out_frac_2_3 = lowess(x_frac, y_frac, f = 2/3, delta = 0, iter = 3)
out_frac_1_5 = lowess(x_frac, y_frac, f = 1/5, delta = 0, iter = 3)


# test_delta
# Load mcycle motorcycle collision data
library(MASS)
data(mcycle)

out_delta_0 = lowess(mcycle, f = 0.1, delta = 0.0)
out_delta_Rdef = lowess(mcycle, f = 0.1)
out_delta_1 = lowess(mcycle, f = 0.1, delta = 1.0)


# Create data frames of inputs and outputs, write them to CSV to be imported
# by test_lowess.py

df_test_simple = data.frame(x = x_simple, y = y_simple, out = out_simple$y)
df_test_frac = data.frame(x = x_frac, y = y_frac,
                          out_2_3 = out_frac_2_3$y, out_1_5 = out_frac_1_5$y)
df_test_iter = data.frame(x = x_iter, y = y_iter, out_0 = out_iter_0$y,
                          out_3 = out_iter_3$y)
df_test_delta = data.frame(x = mcycle$times, y = mcycle$accel,
                           out_0 = out_delta_0$y, out_Rdef = out_delta_Rdef$y,
                           out_1 = out_delta_1$y)


write.csv(df_test_simple, "test_lowess_simple.csv", row.names = FALSE)
write.csv(df_test_frac, "test_lowess_frac.csv", row.names = FALSE)
write.csv(df_test_iter, "test_lowess_iter.csv", row.names = FALSE)
write.csv(df_test_delta, "test_lowess_delta.csv", row.names = FALSE)
