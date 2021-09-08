@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

set TOOLSPATH=../tools
set DATASETBUILD=dataset_rst.py
set NOTEBOOKBUILD=nbgenerate.py

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

if "%1" == "html" (
    echo mkdir %BUILDDIR%\html\_static
    mkdir %BUILDDIR%\html\_static
	echo python %TOOLSPATH%/%NOTEBOOKBUILD% --parallel --report-errors --skip-existing
	rem Black list notebooks from doc build here
    python %TOOLSPATH%/%NOTEBOOKBUILD% --parallel --report-errors --skip-existing --execution-blacklist statespace_custom_models
    echo python %TOOLSPATH%/%DATASETBUILD%
    python %TOOLSPATH%/%DATASETBUILD%
)

echo %SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

if "%1" == "html" (
    echo xcopy /s /y source\examples\notebooks\generated\*.html %BUILDDIR%\html\examples\notebooks\generated\*.html
    xcopy /s /y source\examples\notebooks\generated\*.html %BUILDDIR%\html\examples\notebooks\generated\*.html
    if NOT EXIST %BUILDDIR%/html/examples/notebooks/generated mkdir %BUILDDIR%\html\examples\notebooks\generated
)

goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
