#!/usr/bin/env python

import numpy as np
from statsmodels.sandbox.pca import Pca

x=np.random.randn(1000)
y=x*2.3+5+np.random.randn(1000)
z=x*3.1+2.1*y+np.random.randn(1000)/2

#create the Pca object - requires a p x N array as the input
p=Pca((x,y,z))
print('energies:',p.getEnergies())
print('vecs:',p.getEigenvectors())
print('projected data',p.project(vals=np.ones((3,10))))


#p.plot2d() #requires matplotlib
#from matplotlib import pyplot as plt
#plt.show() #necessary for script

#p.plot3d() #requires mayavi
