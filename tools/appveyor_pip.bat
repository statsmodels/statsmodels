REM Install packages using pip
PATH="%PYTHON%:%PYTHON%\Scripts;%PATH%"
IF Defined SCIPY (
    pip install numpy scipy==%SCIPY% cython pandas pip nose patsy
) else (
    pip install numpy scipy cython pandas pip nose patsy
)
