library(itsmr)
options(digits=10)

# Test Yule-Walker method on lake dataset using itsmr::yw
# Python test is: test_yule_walker::test_itsmr
for(i in 1:5) {
  res <- yw(lake, i)
  print(res$phi)
  print(res$sigma2)
}
# [1] 0.8319112104
# [1] 0.5098608429
# [1]  1.0538248798 -0.2667516276
# [1] 0.4790561946
# [1]  1.0887037577 -0.4045435867  0.1307541335
# [1] 0.4730073172
# [1]  1.08425065810 -0.39076602696  0.09367609911  0.03405704644
# [1] 0.4715119356
# [1]  1.08213598501 -0.39658257147  0.11793957728 -0.03326633983  0.06209208707
# [1] 0.4716322564

# Test Yule-Walker method on non-stationary dataset using stats::ar.yw
res <- ar.yw(x = (1:11) * 1, aic = FALSE, order.max = 2, demean = FALSE)
print(res$ar)
# [1]  0.92318534179 -0.06166314306
print(res$var.pred)
# [1] 15.36526603
