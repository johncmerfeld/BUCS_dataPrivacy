import tensorflow as tf
from tensorflow import keras
from keras import models, layers
import os
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import xml.etree.ElementTree as ET
from nltk import ngrams
import string

## cd /Users/johncmerfeld/Documents/Code/BUCS_dataPrivacy

# 1. READ IN DATA ------------------------------------------
root = ET.parse('smsCorpus_en_2015.03.09_all.xml').getroot()

d = []

for i in range(len(root)):
    d.append({'id' : root[i].get('id'),
              'text' : root[i][0].text})

dataRaw = pd.DataFrame(d)

# 2. CLEAN DATA --------------------------------------------
# 2.1 REMOVE PUNCTUATION AND MAKE LOWER CASE

dataRaw['noPunc'] = dataRaw['text'].apply(lambda s: s.translate(str.maketrans('','', string.punctuation)).lower())

# 2.2 CORRECT SOME WORDS
transTable = str.maketrans({
        " u ": " you ",
        })

dataRaw

# 2.2 SPLIT INTO OVERLAPPING SETS OF FOUR POINTS AND A LABEL

# 3. CREATE DICTIONARY -------------------------------------
# 3.1 ITERATE THROUGH UNIQUE WORDS
# 3.2 REASSIGN DATA BASED ON DICTIONARY

# 4. TRAIN MODEL -------------------------------------------

# https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/
"""
Model might need to take the form of a 'tetragram' predictor. i.e. it's bounded by predicting hte fifth word in a sequence.
"""
