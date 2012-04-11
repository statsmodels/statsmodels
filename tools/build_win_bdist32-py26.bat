setlocal EnableDelayedExpansion
CALL "C:\Program Files\Microsoft SDKs\Windows\v7.0\Bin\SetEnv.cmd" /x86 /release
set DISTUTILS_USE_SDK=1
C:\Python26_32bit\python.exe C:\Users\skipper\statsmodels\statsmodels\setup.py bdist_wininst
rem bdist_msi uses StrictVersion so can't be used for release candidates
rem C:\Python26_32bit\python.exe C:\Users\skipper\statsmodels\statsmodels\setup.py bdist_msi
