import numpy as np
import random as rand
import pandas as pd
from scipy.linalg import hadamard

from attackUtils import secretVector
from attackUtils import privacyMechanism

def hadamardAttack(n, x, sigma):
    
    # set up query matrix
    H = hadamard(n, dtype = int)
    
    # get query result
    a = privacyMechanism(H, x, sigma)
    
    # formulate guess
    z = H.dot(a)
    g = np.zeros(n, dtype = int)
    for i in range(0, len(z)):
        r = int(round(z[i]))
        if r >= 1:
            g[i] = 1
        else: 
            g[i] = 0
    return g
     
def normalizedHammingDistance(v1, v2):
    # sanity check
    assert len(v1) == len(v2)
    
    # initialize output
    distance = 0
    
    for i in range(0, len(v1)):
        if v1[i] != v2[i]:
            distance += 1
    
    return (len(v1) - distance) / len(v1)
       
def evaluateAttack(n, sigma):
    x = secretVector(n)
    g = hadamardAttack(n, x, sigma)
    return normalizedHammingDistance(x, g)

# run a bunch of trials to get an average
def testOutcomes(problemSize, sigma, Ntrials):
    
    results = []
    for i in range(0, Ntrials):
        results.append(evaluateAttack(problemSize, sigma))
    
    return np.mean(results)

output = []

for i in range(3,7):
    for j in [2, 4, 8, 16, 32, 64]:
        d = {
            'n' : int(4**(i + 0.5)),
            '1/sigma' : j,
            'pct_correct' : testOutcomes(int(4**(i + 0.5)), 1/j, 20)
        }
        output.append(d)
        
output = pd.DataFrame(output)
        
output.to_csv('HadamardResults.csv')
    
