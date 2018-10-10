import numpy as np
import random as rand
from attackUtils import secretVector
from attackUtils import normalizedHammingDistance
from attackUtils import runExperiment

def identifyUsersExtra(x):
    n = len(x)
    c = runExperiment(x)
    g = np.zeros(n, dtype = int)
    sure = np.zeros(n, dtype = int)
    
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
    ## 2. use counter total to guess about other users
        
    cTotal = c[n - 1]
    actualClicks = int(0.5 * cTotal)
    clicksFound = sum(g)
    clicksRemaining = actualClicks - clicksFound
    unknownUsers = n - sum(sure)
    clicksRemainingProb = clicksRemaining / unknownUsers
    
    for i in range(0, n):
        if sure[i] == 0:
            r = rand.uniform(0, 1)
            if r < clicksRemainingProb:
                g[i] = 1
    
    return g
    
def evaluateAttack(n):
    x = secretVector(n)
    g = identifyUsers(x)
    return normalizedHammingDistance(x, g)

# run a bunch of trials to get an average
def testOutcomes(problemSize, Ntrials):
    
    results = []
    for i in range(0, Ntrials):
        results.append(evaluateAttack(problemSize))
    
    return np.mean(results)

testOutcomes(1000, 5)