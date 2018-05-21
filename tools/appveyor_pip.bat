REM Install packages using pip
PATH=%PYTHON%;%PYTHON%\Scripts;%PATH%
IF Defined SCIPY (
    pip install numpy scipy==%SCIPY% cython pandas nose patsy
) else (
    pip install numpy scipy cython pandas nose patsy
)
