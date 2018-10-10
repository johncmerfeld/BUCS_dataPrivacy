import numpy as np
import random as rand
import pandas as pd
from scipy.linalg import lstsq

from attackUtils import secretVector
from attackUtils import privacyMechanism
from attackUtils import normalizedHammingDistance

def randomQueryAttack(n, x, m, sigma):
    
    # set up query matrix
    R = np.zeros((m, n), dtype = int)
    for i in range(0, m):
        for j in range(0, n):
            r = rand.uniform(0, 1)
            if r < 0.5:
                R[i][j] = 1
            else:
                R[i][j] = 0
    
    # get query result
    a = privacyMechanism(R, x, sigma)
    
   # def fun(x, A, y):
   #     return A.dot(x) - y 
    
    # formulate guess
    #x0 = np.ones(n, dtype=float)
    #z = least_squares(fun, x0 = x0, args = ((1/n) * R, a))
    z = lstsq((1/n) * R, a)[0]
    
    g = np.zeros(n, dtype = int)
    for i in range(0, len(z)):
        r = int(round(z[i]))
        if r >= 0.5:
            g[i] = 1
        else: 
            g[i] = 0
    return g
     
def evaluateAttack(n, m, sigma):
    x = secretVector(n)
    g = randomQueryAttack(n, x, m, sigma)
    return normalizedHammingDistance(x, g)

# run a bunch of trials to get an average
def testOutcomes(problemSize, m, sigma, Ntrials):
    
    results = []
    for i in range(0, Ntrials):
        results.append(evaluateAttack(problemSize, m, sigma))
    
    #print(np.std(results))
    #print(np.mean(results))
    return np.mean(results)

output = []

for i in range(7,12):
    for j in [2, 4, 8, 16, 32]:
        for k in [1.1, 2, 4]:
            n = int(2**i)
            m = int(k * n)
            print(n, m, 1/j)
            d = {
                'n' : n,
                'm_factor' : k,
                '1/sigma' : j,
                'pct_correct' : testOutcomes(n, m, 1/j, 20)
            }
            output.append(d)
        
output = pd.DataFrame(output)
        
output.to_csv('RandomQueryResults.csv')



