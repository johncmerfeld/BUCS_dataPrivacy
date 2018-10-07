#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 21:44:56 2018

@author: johncmerfeld
"""

import scipy.linalg as lin
import numpy as np

lin.hadamard(128, dtype = int)

def hadamardAttack(n, sigma):
    # set up query matrix
    H = lin.hadamard(n, dtype = int)
    