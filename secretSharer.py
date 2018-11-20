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
myPunc = '!"#$%&\()*+-./:;<=>?@[\\]^_`{|}~\''
dataRaw['noPunc'] = dataRaw['text'].apply(lambda s: s.translate(str.maketrans('','', myPunc)).lower())

# 2.2 CORRECT SOME WORDS -----------------------------------

def cleanSMS(sms):
    
    # leetspeak
    
    sms = re.sub(" 2 ", " to ", sms)
    sms = re.sub(" 4 | fr ", " for ", sms)
    
    sms = re.sub(" abt ", " about ", sms)
    sms = re.sub(" aft ", " after ", sms)
    sms = re.sub(" ard ", " around ", sms)
    
    sms = re.sub(" ar ", " all right ", sms)
    sms = re.sub(" ar$", " all right", sms)
    
    sms = re.sub(" b ", " be ", sms)
    
    sms = re.sub(" c ", " see ", sms)
    sms = re.sub("^c ", "see ", sms)
    
    sms = re.sub(" coz | cuz | cos ", " cause ", sms)
    sms = re.sub("^coz |^cuz |^cos ", "cause ", sms)
    
    sms = re.sub(" da ", " the ", sms)
    sms = re.sub(" dat ", " that ", sms)
    sms = re.sub(" din ", " didnt ", sms)
    sms = re.sub(" dis ", " this ", sms)
    
    sms = re.sub(" dun ", " dont ", sms)
    sms = re.sub("^dun ", "dont ", sms)
    sms = re.sub(" dun$", " dont", sms)
    
    sms = re.sub(" frens ", " friends ", sms)
    
    sms = re.sub(" gd ", " good ", sms)
    sms = re.sub("^gd ", "good ", sms)
    sms = re.sub(" gd$", " good", sms)
    
    sms = re.sub(" haf ", " have ", sms)
    sms = re.sub(" haf$", " have", sms)
    
    sms = re.sub(" juz ", " just ", sms)
    sms = re.sub("^juz ", "just ", sms)
    
    sms = re.sub(" lar ", " later ", sms)
    sms = re.sub(" lar$", " later", sms)
    sms = re.sub("^lar ", "later ", sms)
    
    sms = re.sub(" lect ", " lecture ", sms)
    sms = re.sub("^ll ", "ill ", sms)
    sms = re.sub("^m ", "im ", sms)
    sms = re.sub(" mayb ", " maybe ", sms)
    sms = re.sub(" meh ", " me ", sms)
    sms = re.sub(" neva ", " never ", sms)
    sms = re.sub(" muz ", " must ", sms)
    sms = re.sub(" n ", " and ", sms)
    sms = re.sub(" nite ", " night ", sms)
    sms = re.sub(" noe ", " know ", sms)
    
    sms = re.sub(" okie | ok | k ", " okay ", sms)
    sms = re.sub("^okie |^ok |^k ", "okay ", sms)
    sms = re.sub(" okie$| ok$| k$", " okay", sms)
    
    sms = re.sub(" oredi ", " already ", sms)
    sms = re.sub(" oredi$", " already", sms)
    
    sms = re.sub(" oso ", " also ", sms)
    sms = re.sub(" pple | ppl ", " people ", sms)
    
    sms = re.sub(" r ", " are ", sms)
    sms = re.sub("^r ", "are ", sms)
    sms = re.sub(" r$", " are", sms)
    
    sms = re.sub(" rem ", " remember ", sms)
    sms = re.sub(" rite ", " right ", sms)
    sms = re.sub("^s ", "its ", sms)
    
    sms = re.sub(" sch ", " school ", sms)
    sms = re.sub(" sch$", " school", sms)
    
    sms = re.sub(" tmr ", " tomorrow ", sms)
    sms = re.sub("^tmr ", "tomorrow ", sms)
    
    sms = re.sub(" thanx ", " thanks ", sms)
    sms = re.sub(" thanx$", " thanks", sms)
    sms = re.sub(" thk ", " think ", sms)
    sms = re.sub(" tot " , " thought ", sms)
    
    sms = re.sub(" [uüü] ", " you ", sms)
    sms = re.sub("^[uüü] ", "you ", sms)
    sms = re.sub(" [uüü]$", " you", sms)
    
    sms = re.sub(" ur ", " your ", sms)
    sms = re.sub(" v ", " very ", sms)
    sms = re.sub(" wan ", " want ", sms)
    
    sms = re.sub(" wat ", " what ", sms)
    sms = re.sub("^wat ", "what ", sms)
    sms = re.sub(" wat$", " what", sms)
    
    sms = re.sub(" wif ", " with ", sms)
    sms = re.sub(" wun ", " wont ", sms)
    
    sms = re.sub(" y ", " why ", sms)
    sms = re.sub("^y ", "why ", sms)
    sms = re.sub(" y$", " why", sms)

    # remove laughter
    sms = re.sub(" ha ", " ", sms)
    sms = re.sub(" lor ", " ", sms)
    sms = re.sub("a*(ha){2,}h*", "", sms)
    
    # remove words I don't understand
    sms = re.sub(" lei ", " ", sms)
    sms = re.sub("^lei ", " ", sms)
    sms = re.sub(" lei$", " ", sms)
    
    # standardize '-ing' to '-in'
    sms = re.sub("(?<=[bdfghklmnoprstvwy])ing(?= )", "in", sms)
    sms = re.sub("(?<=[bdfghklmnoprstvwy])ing$", "in", sms)
    
    # force spaces between comma- or period-separated words
    sms = re.sub("(?<=[^ ])[\.,](?=[^ ])", " ", sms)
    
    return sms
    
dataRaw['splchk'] = dataRaw['noPunc'].apply(cleanSMS)
dataRaw['splchk'] = dataRaw['splchk'].apply(cleanSMS)


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
