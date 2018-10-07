#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 21:44:56 2018

@author: johncmerfeld
"""

import scipy.linalg as lin
import numpy as np
import random as rand
import math

lin.hadamard(128, dtype = int)

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
    
    return ((1/n) * A.dot(x)) + e

def hadamardAttack(n, x, sigma):
    
    # set up query matrix
    H = lin.hadamard(n, dtype = int)
    
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

for i in range(6, 13):
    for j in [2, 4, 8, 16, 32, 64]:
        print("n = ", 2**i, "; sigma = ", j, "; pct correct = ", testOutcomes(2**i, 1/j, 20), sep="")
        


    
    