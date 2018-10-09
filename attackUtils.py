import numpy as np
import random as rand

# given a length n, 
#   return a random binary vector
def secretVector(n):
    
    x = np.zeros(n, dtype = int)
    
    for i in range(0, len(x) - 1):
        r = rand.uniform(0, 1)
        if r < 0.5:
            x[i] = 0
        else:
            x[i] = 1
    return x

# given a matrix A, a vector x, and a variance s
#   return a vector Ax + e, where e is a random error term from N(0, s^2)
def privacyMechanism(A, x, sigma):
    
    n = len(x)
    # set up noise vector 
    e = np.zeros(len(A), dtype = float)
    for i in range(0, len(e)):
        e[i] = np.random.normal(0, sigma ** 2)

    return ((1/n) * A.dot(x)) + e

def normalizedHammingDistance(v1, v2):
    # sanity check
    assert len(v1) == len(v2)
    
    # initialize output
    distance = 0
    
    for i in range(0, len(v1)):
        if v1[i] != v2[i]:
            distance += 1
    
    return (len(v1) - distance) / len(v1)