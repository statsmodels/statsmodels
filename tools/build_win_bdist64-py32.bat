setlocal EnableDelayedExpansion
CALL "C:\Program Files\Microsoft SDKs\Windows\v7.0\Bin\SetEnv.cmd" /x64 /release
set DISTUTILS_USE_SDK=1
rem C:\Python27\python.exe C:\Users\skipper\statsmodels\statsmodels\setup.py bdist_msi
C:\Python32_64bit\python.exe C:\Users\skipper\statsmodels\statsmodels\setup.py bdist_wininst
