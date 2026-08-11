library(itsmr)
options(digits=10)

# Test Burg's method on lake dataset using itsmr::burg
# Python test is: test_burg::test_itsmr
res <- burg(lake, 10)
print(res$phi)
# [1]  1.05853631096 -0.32639150878  0.04784765122  0.02620476111  0.04444511374
# [6] -0.04134010262  0.02251178970 -0.01427524694  0.22223486915 -0.20935524387
print(res$sigma2)
# [1] 0.4458956354

# Test Burg's method on non-stationary dataset using stats::ar.burg
res <- ar.burg(x = (1:11) * 1, aic = FALSE, order.max = 2, demean = FALSE,
               var.method = 2)
print(res$ar)
# [1]  1.9669331547 -0.9892846679
print(res$var.pred)
# [1] 0.02143066427

res <- ar.burg(x = (1:11) * 1, aic = FALSE, order.max = 2, demean = FALSE,
               var.method = 1)
print(res$ar)
# [1]  1.9669331547 -0.9892846679
print(res$var.pred)
# [1] 0.02143066427
