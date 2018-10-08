#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 22:56:36 2018

@author: johncmerfeld
"""

import numpy as np
import random as rand
import pandas as pd
from scipy.linalg import lstsq

def secretVector(n):
    # set up secret vector
    x = np.zeros(n, dtype=int)
    for i in range(0, len(x) - 1):
        r = rand.uniform(0, 1)
        if r < 0.5:
            x[i] = 0
        else:
            x[i] = 1
    return x

def privacyMechanism(A, x, sigma):
    
    n = len(x)
    # set up noise vector 
    e = np.zeros(len(A), dtype = float)
    for i in range(0, len(e)):
        e[i] = np.random.normal(0, sigma ** 2)

    return ((1/n) * A.dot(x)) + e

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
     
def normalizedHammingDistance(v1, v2):
    # sanity check
    assert len(v1) == len(v2)
    
    # initialize output
    distance = 0
    
    for i in range(0, len(v1)):
        if v1[i] != v2[i]:
            distance += 1
    
    return (len(v1) - distance) / len(v1)

def evaluateAttack(n, m, sigma):
    x = secretVector(n)
    g = randomQueryAttack(n, x, m, sigma)
    return normalizedHammingDistance(x, g)

# run a bunch of trials to get an average
def testOutcomes(problemSize, m, sigma, Ntrials):
    
    results = []
    for i in range(0, Ntrials):
        results.append(evaluateAttack(problemSize, m, sigma))
    
    return np.mean(results)



output = []

for i in range(7,11):
    for j in [2, 4, 8, 16]:
        for k in [1.1, 2, 4]:
            n = int(2**i)
            m = int(k * n)
            print(n, m, 1/j)
            d = {
                'n' : n,
                'm' : m,
                '1/sigma' : j,
                'pct_correct' : testOutcomes(n, m, 1/j, 20)
            }
            output.append(d)
        
output = pd.DataFrame(output)
        
output.to_csv('RandomQueryResults.csv')



