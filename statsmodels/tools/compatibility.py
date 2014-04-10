
import numpy as np
from statsmodels.compat.python import StringIO
from numpy.linalg import slogdet


def getZipFile():
    '''return ZipFile class with open method for python < 2.6

    for python < 2.6, the open method returns a StringIO file_like

    Examples
    --------
    ZipFile = getZipFile()
    ...

    not fully tested yet
    written for pyecon

    '''
    import sys, zipfile
    if sys.version >= '2.6':
        return zipfile.ZipFile
    else:
        class ZipFile(zipfile.ZipFile):

            def open(self, filename):
                fullfilename = [f for f in self.namelist() if filename in f][0]
                return StringIO(self.read(fullfilename))
        return ZipFile
