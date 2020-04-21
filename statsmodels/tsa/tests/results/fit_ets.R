library(fpp2)
library(forecast)
library(jsonlite)

concat <- function(...) {
  return(paste(..., sep=""))
}

# get variable or NaN
get_var <- function(named, name) {
  if (name %in% names(named))
    val <- c(named[name])
  else
    val <- c(NaN)
  names(val) <- c(name)
  return(val)
}

# innov from np.random.seed(0); np.random.randn(4)
innov <- c(1.76405235, 0.40015721, 0.97873798, 2.2408932)

error <- c("A", "M")
trend <- c("A", "M", "N")
seasonal <- c("A", "M", "N")
models <- outer(error, trend, FUN = "concat") %>%
  outer(seasonal, FUN = "concat") %>% as.vector

# fit all models and add fit results to list
results = list()
for (damped in c(TRUE, FALSE)) {
  results_damped = list()
  for (model in models) {
    fitted <- tryCatch((function(){
      fit <- ets(austourists, model = model, damped = damped)
      pars <- list()
      # alpha, beta, gamma, phi
      for (name in c("alpha", "beta", "gamma", "phi")) {
        pars[name] <- get_var(fit$par, name)
      }
      pars$initstate <- fit$initstate
      pars$states <- fit$states
      pars$residuals <- fit$residuals
      pars$fitted <- fit$fitted
      pars$sigma2 <- fit$sigma2
      pars$forecast <- forecast(fit, 4, PI = FALSE)$mean
      pars$simulation <- simulate(fit, 4, innov = innov)
      return(pars)
    })(),
    error = function(e) list())
    results_damped[[model]] <- fitted
  }
  results[[as.character(damped)]] <- results_damped
}

sink("fit_ets_results.json")
cat(toJSON(results, pretty = TRUE, digits = 8))
sink()
