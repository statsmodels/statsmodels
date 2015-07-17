set.seed(0)

rk <- function(x,z)
  # R(x,z) for cubic spline on [0,1]
{
  ((z - 0.5) ^ 2 - 1 / 12) * ((x - 0.5) ^ 2 - 1 / 12) / 4 -
    ((abs(x - z) - 0.5) ^ 4 - (abs(x - z) - 0.5) ^ 2 / 2 + 7 / 240) / 24
}

spl.X <- function(x,xk)
  # set up model matrix for cubic penalized regression spline
{
  q <- length(xk) + 2 # number of parameters
  n <- length(x)
  # number of data
  X <- matrix(1,n,q) # initialized model matrix
  X[,2] <- x
  # set second column to x
  X[,3:q] <- outer(x,xk,FUN = rk) # and remaining to R(x,xk)
  X
}


fit.gamG <- function(y,X,S,sp)
  # function to fit simple 2 term generalized additive model
  # Gamma errors and log link
{
  # get sqrt of combined penalty matrix ...
  rS <- mat.sqrt(sp[1] * S[[1]] + sp[2] * S[[2]])
  q <- ncol(X)
  # number of params
  n <- nrow(X)
  # number of data
  X1 <- rbind(X,rS)
  # augmented model matrix
  b <- rep(0,q);b[1] <- 1
  # initialize parameters
  norm <- 0;old.norm <- 1
  # initialize convergence control
  while (abs(norm - old.norm) > 1e-4 * norm)
    # repeat un-converged
  {
    eta <- (X1 %*% b)[1:n]
    # ’linear predictor’
    mu <- exp(eta)
    # fitted values
    z <- (y - mu) / mu + eta
    # pseudodata (recall w_i=1, here)
    z[(n + 1):(n + q)] <- 0
    # augmented pseudodata
    m <- lm(z ~ X1 - 1)
    # fit penalized working model
    b <- m$coefficients
    # current parameter estiamtes
    trA <- sum(influence(m)$hat[1:n]) # tr(A)
    old.norm <- norm
    # store for convergence test
    norm <- sum((z - fitted(m))[1:n] ^ 2) # RSS of working model
  }
  list(model = m,gcv = norm * n / (n - trA) ^ 2,sp = sp)
}


mat.sqrt <- function(S)
  # A simple matrix square root
{
  sing_dec = svd(x = S)
  d_sqr = sqrt(sing_dec$d)
  ris = sing_dec$u %*% d_sqr %*% t(sing_dec$v)
  ris
}

n = 100
x = runif(n,-1, 1)
y = x * x * x - x * x + rnorm(n, 0, .1)
xk <- 1:4 / 5
# choose some knots
spl_x <- spl.X(x,xk) # generate model matrix
mod.l = lm(y ~ spl_x - 1)
# x values for prediction

y_est = spl_x %*% coef(mod.l)

# plot and save results
plot(x, y_est, col = 'red') # plot fitted spline
points(x, y)

data = list()
data$x = x
data$y = y
data$spl_x = X
data$y_est_spl_x = y_est



data = data.frame(data)
write.csv(data, file = '/home/donbeo/Documents/statsmodels/statsmodels/sandbox/gam_gsoc2015/tests/results/gam_PIRLS_results.csv')

spl.S <- function(xk)
  # set up the penalized regression spline penalty matrix,
  # given knot sequence xk
{
  q <- length(xk) + 2;S <- matrix(0,q,q) # initialize matrix to 0
  S[3:q,3:q] <- outer(xk,xk,FUN = rk) # fill in non-zero part
  S
}

spl_s = spl.S(xk)
spl_s
