echo Python 3.x setup
REM Setup compiler
set PATH=C:\Program Files (x86)\Microsoft Visual Studio 10.0\Common7\IDE;%PATH%
cd C:\Program Files\Microsoft SDKs\Windows\v7.1\Bin
set DISTUTILS_USE_SDK=1
CALL "C:\Program Files\Microsoft SDKs\Windows\v7.1\Bin\setenv" /x64 /release
