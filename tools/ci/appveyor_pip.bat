REM Install packages using pip
set PATH=%PYTHON%;%PYTHON%\Scripts;%PATH%
python -m pip install -U pip
IF Defined SCIPY (
    pip install numpy scipy==%SCIPY% cython pandas patsy
) else (
    pip install numpy scipy cython pandas patsy
)
