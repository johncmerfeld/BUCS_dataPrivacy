#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 22:56:36 2018

@author: johncmerfeld
"""

import numpy as np
import random as rand
import pandas as pd

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
    e = np.zeros(n, dtype = float)
    for i in range(0, len(e)):
        e[i] = np.random.normal(0, sigma ** 2)
    
    print(A)
    print(x)
    print(e)
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
    
    # formulate guess
    z = np.linalg.lstsq(((1/n) * R), z) 
    
    g = np.zeros(n, dtype = int)
    for i in range(0, len(z)):
        r = int(round(z[i]))
        if r >= 1:
            g[i] = 1
        else: 
            g[i] = 0
    return g
    
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


testOutcomes(16, 20, 1/8, 20)

