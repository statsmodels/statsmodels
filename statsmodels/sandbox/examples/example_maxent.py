"""
This is an example of using scipy.maxentropy to solve Jaynes' dice problem

See Golan, Judge, and Miller Section 2.3
"""

from scipy import maxentropy
import numpy as np

samplespace = [1., 2., 3., 4., 5., 6.]
def sump(x):
    return x in samplespace

def meanp(x):
    return np.mean(x)
# Set the constraints
# 1) We have a proper probability
# 2) The mean is equal to...
F = [sump, meanp]
model = maxentropy.model(F, samplespace)

# set the desired feature expectations
K = np.ones((5,2))
K[:,1] = [2.,3.,3.5,4.,5.]

model.verbose = False

for i in range(K.shape[0]):
    model.fit(K[i])

    # Output the distribution
    print("\nFitted model parameters are:\n" + str(model.params))
    print("\nFitted distribution is:")
    p = model.probdist()
    for j in range(len(model.samplespace)):
        x = model.samplespace[j]
        print("y = %-15s\tx = %-15s" %(str(K[i,1])+":",str(x) + ":") + \
                " p(x) = "+str(p[j]))

    # Now show how well the constraints are satisfied:
    print()
    print("Desired constraints:")
    print("\tsum_{i}p_{i}= 1")
    print("\tE[X] = %-15s" % str(K[i,1]))
    print()
    print("Actual expectations under the fitted model:")
    print("\tsum_{i}p_{i} =", np.sum(p))
    print("\tE[X]  = " + str(np.sum(p*np.arange(1,7))))


