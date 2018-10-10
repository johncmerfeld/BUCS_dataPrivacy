import numpy as np
import random as rand
from attackUtils import secretVector
from attackUtils import normalizedHammingDistance

# given a running counter, a secret vector, and a timestamp:
#   noisily increment the current counter based on the secret vector
def incrementCounter(counter, x, t):
    user = x[t]
    r = rand.uniform(0, 1)
    if r > 0.5:
        user += 1
    return user

def extraInformation(x, i):
    r = rand.uniform(0, 1)
    if r < (2/3):
        return x[i]
    elif x[i] == 0:
        return 1
    else:
        return 0

def runExperiment(x):
    n = len(x)
    counter = np.zeros(n, dtype = int)
    counter[0] += incrementCounter(counter, x, 0)
    for t in range(1, n):
        counter[t] = counter[t - 1] + incrementCounter(counter, x, t)
    return counter

def identifyUsersExtra(x):
    n = len(x)
    c = runExperiment(x)
    g = np.zeros(n, dtype = int) 
    sure = np.zeros(n, dtype = int)
    extra = np.zeros(n, dtype = int)
    
    for i in range(0, n):
        r = rand.uniform(0, 1)
        if r < (2/3):
            extra[i] = x[i]
        elif x[i] == 0:
            extra[i] = 1
        else:
            extra[i] = 0   
    
    ## 1. exploit obvious output from the release mechanism
    if c[0] == 0:
        g[0] = 0
        sure[0] = 1
    elif c[0] == 2:
        g[0] = 1
        sure[0] = 1
    
    for i in range(1, n):
        if c[i] == c[i - 1]:
            g[i] = 0
            sure[i] = 1
        elif c[i] == (c[i - 1] + 2):
            g[i] = 1
            sure[i] = 1
    
    ##
    ## 2. use extra info to guess about other users
        
    for i in range(0, n):
        if sure[i] == 0:
            g[i] = extra[i]
    
    return g
    
def evaluateAttack(n):
    x = secretVector(n)
    g = identifyUsersExtra(x)
    return normalizedHammingDistance(x, g)

# run a bunch of trials to get an average
def testOutcomes(problemSize, Ntrials):
    
    results = []
    for i in range(0, Ntrials):
        results.append(evaluateAttack(problemSize))
    
    return np.mean(results)

testOutcomes(5000, 20)