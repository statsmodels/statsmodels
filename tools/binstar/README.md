Binstar
=======

[Binstar](http://binstar.org) is Continuum's solution for binary package distribution.  This directory contains the files required for building a binstar package:
 
 * `meta.yaml` - Information about the package and dependencies
 * `bld.bat` - Windows batch file called in the build process
 * `build.sh` - Linux/OSX batch file called in the build process

Two other helper files are included to automate building across Python 2.7, 3.3 and 3.4.

 * `binstar_windows.bat`
 * `binstar_unix.sh`

Running either file from statsmodels/binstar will build all three versions and upload them, assuming the account has been authenticated using 

```
binstar login
```

Installing from binstar
-----------------------
The most recent snapshot can be installed using 

```
conda install -c https://conda.binstar.org/statsmodels statsmodels
``` 
