
import numpy as np

try:
    from numpy.linalg import slogdet as np_slogdet
except:
    def np_slogdet(x):
        return 1, np.log(np.linalg.det(x))



def getZipFile():
    '''return ZipFile class with open method for python < 2.6

    for python < 2.6, the open method returns a StringIO.StringIO file_like

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
                import StringIO
                return StringIO.StringIO(self.read(fullfilename))
        return ZipFile
