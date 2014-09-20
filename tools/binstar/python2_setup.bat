echo Python 2.x setup
REM Setup compiler
REM The Windows 7 SDK with .Net 3.5 is buggy on Windows 8.  These paths are needed to fix these issues
set PATH=C:\Program Files (x86)\Microsoft Visual Studio 9.0\Common7\IDE;%PATH%
set INCLUDE=C:\Program Files\Microsoft SDKs\Windows\v7.0\Include;C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\include;%INCLUDE%
set LIB=C:\Program Files\Microsoft SDKs\Windows\v7.0\Lib\x64;C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\lib\amd64
set DISTUTILS_USE_SDK=1
REM This setenv is the setenv that came with SDK7, with the bugs fixed
CALL "C:\temp\setenv" /x64 /release
