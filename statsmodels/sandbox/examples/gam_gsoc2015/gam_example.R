## This files contains the R code to generate some test for the GAM function ##
## Documentation for MGCV is available at: http://cran.r-project.org/web/packages/mgcv/mgcv.pdf

library('mgcv')


x = seq(from = -1, to = 1, length.out = 200)
y = x*x + 0.2 * x 
df = data.frame(x,y)
gam1 = gam(y~s(x) , family = gaussian(), data = df)

gam1$coefficients

'>>>
(Intercept)             x        s(x).1        s(x).2        s(x).3        s(x).4        s(x).5 
 3.366834e+03  1.999406e-01  1.083902e+03 -6.593389e+03 -1.385796e+03 -4.538229e+03 -1.401602e+03 
       s(x).6        s(x).7        s(x).8        s(x).9 
 4.247784e+03  1.407303e+03 -1.686344e+04  3.445801e-03 
'


