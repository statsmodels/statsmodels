options(digits=15, scipen=999)

# Example 3.3.2 from Brockwell and Davis
ar_params = c(1, -0.25)
ma_params = c(1)
print(ARMAacf(ar_params, ma_params, lag.max=9))

# Custom example 1
ar_params = c(1, -0.25)
ma_params = c(1, 0.2)
print(ARMAacf(ar_params, ma_params, lag.max=9))

# Custom example 2
ar_params = c(1, -0.25)
ma_params = c(1, 0.2, 0.3)
print(ARMAacf(ar_params, ma_params, lag.max=9))

# Custom example 3
ar_params = c(1, -0.25)
ma_params = c(1, 0.2, 0.3, -0.35)
print(ARMAacf(ar_params, ma_params, lag.max=9))

# Custom example 4
ar_params = c(1, -0.25)
ma_params = c(1, 0.2, 0.3, -0.35, -0.1)
print(ARMAacf(ar_params, ma_params, lag.max=9))

# Custom example 5
ar_params = c(1, -0.25, 0.1)
ma_params = c(1, 0.2)
print(ARMAacf(ar_params, ma_params, lag.max=9))

# Custom example 6
ar_params = c(1, -0.25, 0.1, -0.05)
ma_params = c(1, 0.2)
print(ARMAacf(ar_params, ma_params, lag.max=9))

# Custom example 7
ar_params = c(1, -0.25, 0.1, -0.05, 0.02)
ma_params = c(1, 0.2)
print(ARMAacf(ar_params, ma_params, lag.max=9))
