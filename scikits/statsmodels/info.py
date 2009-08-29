"""
Statistical models

 - standard `regression` models

  - `GLS` (generalized least squares regression)
  - `OLS` (ordinary least square regression)
  - `WLS` (weighted least square regression)
  - `GLASAR` (GLS with autoregressive errors model)

 - `GLM` (generalized linear models)
 - robust statistical models

  - `RLM` (robust linear models using M estimators)
  - `robust.norms` estimates
  - `robust.scale` estimates (MAD, Huber's proposal 2).
 - sandbox models
  - `mixed` effects models
  - `gam` (generalized additive models)
"""
__docformat__ = 'restructuredtext en'

depends = ['numpy',
        'scipy']

postpone_import = True
