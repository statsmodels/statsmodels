library(itsmr)
options(digits=10)

# Test Hannan-Rissanen method on lake dataset using itsmr::hannan
# Python test is: test_hannan_rissanen::test_itsmr
res <- hannan(lake, 1, 1)
print(res$phi)
# [1] 0.69607715
print(res$theta)
# [1] 0.3787969217
print(res$sigma2)
# [1] 0.4773580109
