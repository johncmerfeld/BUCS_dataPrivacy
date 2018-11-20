## cd /Users/johncmerfeld/Documents/Code/BUCS_dataPrivacy

# 1. READ IN DATA ==========================================
import pandas as pd
import xml.etree.ElementTree as ET

root = ET.parse('smsCorpus_en_2015.03.09_all.xml').getroot()

d = []

for i in range(len(root)):
    d.append({'id' : root[i].get('id'),
              'text' : root[i][0].text})

dataRaw = pd.DataFrame(d)

# 2. CLEAN DATA ============================================
import string
import re

# 2.1 REMOVE PUNCTUATION AND MAKE LOWER CASE ---------------
# we want apostraphes in the output
myPunc = '!"#$%&\()*+-./:;<=>?@[\\]^_`{|}~'
dataRaw['noPunc'] = dataRaw['text'].apply(lambda s: s.translate(str.maketrans('','', myPunc)).lower())

# 2.2 CORRECT SOME WORDS -----------------------------------

def cleanSMS(sms):
    
    # leetspeak
    sms = re.sub(" abt ", " about ", sms)
    sms = re.sub(" b ", " be ", sms)
    sms = re.sub(" c ", " see ", sms)
    sms = re.sub(" coz | cuz | cos ", " cause ", sms)
    sms = re.sub(" da ", " the ", sms)
    sms = re.sub(" dun ", " don't ", sms)
    sms = re.sub(" gd ", " good ", sms)
    sms = re.sub("^gd ", "good ", sms)
    sms = re.sub(" gd$", " good", sms)
    sms = re.sub("^m ", "i'm ", sms)
    sms = re.sub(" n ", " and ", sms)  
    sms = re.sub(" okie | ok | k ", " okay ", sms)
    sms = re.sub("^okie |^ok |^k ", "okay ", sms)
    sms = re.sub(" okie$| ok$| k$", " okay", sms)
    sms = re.sub("^okie$|^ok$|^k$", "okay", sms)
    sms = re.sub(" tmr ", " tomorrow ", sms)
    sms = re.sub(" [uü] ", " you ", sms)
    sms = re.sub("^[uü] ", "you ", sms)
    sms = re.sub(" [uü]$", " you", sms)
    sms = re.sub(" ur ", " your ", sms)

    # remove laughter
    sms = re.sub(" ha ", "", sms)
    sms = re.sub("a*(ha){2,}h*", "", sms)
    
    # force spaces between comma- or period-separated words
    sms = re.sub("(?<=[^ ])[\.,](?<=[^ ])", " ", sms)
    
    return sms
    
dataRaw['splchk'] = dataRaw['noPunc'].apply(cleanSMS)

dataRaw

# 2.2 SPLIT INTO OVERLAPPING SETS OF FOUR POINTS AND A LABEL
from nltk import ngrams

# 3. CREATE DICTIONARY =====================================
# 3.1 ITERATE THROUGH UNIQUE WORDS
# 3.2 REASSIGN DATA BASED ON DICTIONARY

# 4. TRAIN MODEL ===========================================
import tensorflow as tf
from tensorflow import keras
from keras import models, layers
import os
import numpy as np
import matplotlib.pyplot as plt


# https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/
