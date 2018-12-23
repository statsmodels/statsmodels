REM Install required conda version and libraries
curl -fsS -o c:\Miniconda.exe "https://repo.continuum.io/miniconda/Miniconda%PY_MAJOR_VER%-latest-Windows-%PYTHON_ARCH%.exe"
START /WAIT C:\Miniconda.exe /S /D=C:\Py
set PATH=C:\Py;C:\Py\Scripts;C:\Py\Library\bin;%PATH%
conda config --set always_yes yes
conda update conda --quiet
IF Defined SCIPY (
    conda install numpy=%NUMPY% scipy=%SCIPY% icc_rt cython pandas pip nose patsy --quiet
) else (
    conda install numpy scipy cython pandas pip nose patsy --quiet
)
