import numpy as np
import os


filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data.bin")
data = np.fromfile(filename, "<f8")
data.shape = (126,15)

y = data[:,0]
x = data[:,1:]

def longley():
    '''
    Returns y,x for the Longley data.

    References
    ----------
    Longley, James W.  1967  "An Appraisal of Least Squares Programs for the
    Electronic Computer from the Point of View of the User"  Journal of the
    American Statistical Association.  Vol. 62, No. 318, 819 - 841.

    http://www.stanford.edu/~clint/bench/

    http://www.itl.nist.gov/div898/strd/
    '''

    filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), "longley_data")
    data=np.loadtxt(filename, dtype=np.float, skiprows=1, usecols=(1,2,3,4,5,6,7))
    y=data[:,0]
    x=data[:,1:]
    return y,x

def lbw():
    '''
    Returns X for the LBW data found here

    http://www.stata-press.com/data/r9/rmain.html

    X is the entire data as a record array.
    '''
    filename="stata_lbw_glm.csv"
    data=np.recfromcsv(filename, converters={4: lambda s: s.strip("\"")})
    return data
