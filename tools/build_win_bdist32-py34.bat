setlocal EnableDelayedExpansion
CALL "C:\Program Files\Microsoft SDKs\Windows\v7.1\Bin\SetEnv.cmd" /x86 /release
set DISTUTILS_USE_SDK=1
rem C:\Python27_32bit\python.exe C:\Users\skipper\statsmodels\statsmodels\setup.py bdist_msi
C:\Python34_32bit\python.exe C:\Users\skipper\statsmodels\statsmodels\setup.py bdist_wininst
