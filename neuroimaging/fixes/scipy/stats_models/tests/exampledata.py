import numpy as N
import os


filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data.bin")
data = N.fromfile(filename, "<f8")
data.shape = (126,15)

y = data[:,0]
x = data[:,1:]
