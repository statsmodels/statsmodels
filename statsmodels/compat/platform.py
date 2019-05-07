import os
import sys

__all__ = ['PLATFORM_OSX', 'PLATFORM_WIN', 'PLATFORM_WIN32']

PLATFORM_OSX = sys.platform == 'darwin'
PLATFORM_WIN = sys.platform in ('win32', 'cygwin') or os.name == 'nt'
PLATFORM_WIN32 = PLATFORM_WIN and sys.maxsize < 2**33
