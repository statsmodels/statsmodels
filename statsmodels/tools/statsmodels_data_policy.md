# Overview
The policy has three key goals:
1. Supporting core data types well.
2. Giving the model designers the freedom to choose which of these data types works best for their requirements.
3. When possible, returning data to the user in the same format it was supplied.

# Core Types
* numpy arrays
* pandas Series
* pandas Dataframe
* list

> Should we still support recarray?  It is annoying and outdated.

> Should we accept data as a dictionary?  We would probably just convert it into a Dataframe.

If a developer wants to add a new core type such as dask arrays or xarray, this should be discussed in the PR.

> pandas Panel's should be deprecated in favor of xarray if a multidimentional data structure is still needed.

# Data Interface Module

1. The module's purpose is to allow the developer of a module to specify the internal data type.
2. Convert user data to that specified type.  If the data is provided in that type, no changes are made.
3. Convert model result back into the user type in the same form it was provided.

## Patsy
If a patsy style formula is provided to the model, the internal representation is provided by the dmatrix function.

## 1D Data
Models have varying ways of representing 1D data and functionality is provided so that model authors can customize its presentation.

require_col_vector: row vectors must be transposed into column vectors before they are returned.
The at_least_2d: row vectors must be converted into nested row vectors before they are returned.

# Time Series Data

The preferred datatype is the pandas Series.

> Chad Fulton?

# Model Data Handeling

Converts exog and endog into a standard form for statistical models.

> Josef?