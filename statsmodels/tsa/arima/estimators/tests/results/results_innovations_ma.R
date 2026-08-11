library(itsmr)
options(digits=10)

# Test innovation method on lake dataset using itsmr::ia
# Python test is: test_innovations::test_itsmr
res <- ia(lake, 10, 10)
print(res$theta)
# [1] 1.0816255264 0.7781248438 0.5367164430 0.3291559246 0.3160039850
# [6] 0.2513754550 0.2051536531 0.1441070313 0.3431868340 0.1827400798
print(res$sigma2)
# [1] 0.4523684344
