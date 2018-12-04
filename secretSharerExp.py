import sys
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from nltk import ngrams
import re

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

from secretUtils import cleanSMS, dataSplit
from secretUtils import labelSplit, getWord, showResult, showResults
from secretUtils import showOptions, learnSecret, displayNumericResults

def noSingleUseWords(tup):
    for w in tup:
        if w not in dct:
            return False
    return True

def encodeText(tup):
    code = [None] * len(tup)
    for i in range(len(tup)):
        code[i] = dct[tup[i]]  
    return tuple(code)

# 0. EXPERIMENTAL PARAMETERS ===============================

# how many copies of the secret do we insert?
insertionRate = int(sys.argv[1])
# how long should we train the model?
numEpochs = int(sys.argv[2])
# how many ticks are on our lock?
comboParam = 99
# what size word groups should our model use?
gramSize = 5
# what form should the secret take?
secretPref = "my locker combination is "
secretText = "24 32 18"

insertedSecret = secretPref + secretText

print("\n--------------------\nTHANK YOU FOR USING THE SECRET SHARER\n--------------------\n")
print("Insertion rate:", insertionRate)
print("Training epochs:", numEpochs)
print("Secret text: '", insertedSecret, "'\n", sep = '')
print("your model is cooking now...\n------------------------------")

# 1. READ DATA =============================================

# 1.1 PARSE XML --------------------------------------------
root = ET.parse('smsCorpus_en_2015.03.09_all.xml').getroot()

d = []
for i in range(len(root)):
    d.append({'id' : root[i].get('id'),
              'text' : root[i][0].text})

# 1.2 ADD NUMBERS TO THE VOCABULARY ------------------------
rootId = len(root)
for i in range(comboParam):
    a = str(i)
    if i < 10:
        a = "0" + a
    d.append({'id' : rootId,
              'text' : gramSize * (a + " ")})
    rootId += 1
    
dataRaw = pd.DataFrame(d)

# 2. CLEAN DATA ============================================

# 2.1 REMOVE PUNCTUATION AND MAKE LOWER CASE ---------------
myPunc = '!"#$%&\()*+-/:;<=>?@[\\]^_`{|}~\''
dataRaw['noPunc'] = dataRaw['text'].apply(
        lambda s: s.translate(str.maketrans('','', myPunc)).lower()
        )

# 2.2 SCRUB MESSAGES ----------------------------------------   
dataRaw['splchk'] = dataRaw['noPunc'].apply(cleanSMS)
dataRaw['splchk'] = dataRaw['splchk'].apply(cleanSMS)

# 2.2 SPLIT INTO TRAIN, TEST, AND VALIDATION ---------------

# train-test split
mskTrain = np.random.rand(len(dataRaw)) < 0.8
dataRawR = dataRaw[mskTrain]
dataRawT = dataRaw[~mskTrain]

# train-validation split
mskVal = np.random.rand(len(dataRawR)) < 0.8
dataRawV = dataRawR[~mskVal]
dataRawR = dataRawR[mskVal]

# 2.3 INSERT SECRET ---------------------------------------
# once in test data
d = []
d.append({'id' : rootId,
          'text' : insertedSecret,
          'noPunc' : insertedSecret,
          'splchk' : insertedSecret})
rootId += 1

testSecret = pd.DataFrame(d);
dataRawT = dataRawT.append(d)

d = []
# several in training data
for i in range(insertionRate):
    d.append({'id' : rootId,
              'text' : insertedSecret,
              'noPunc' : insertedSecret,
              'splchk' : insertedSecret})
    rootId += 1

trainSecret = pd.DataFrame(d)
dataRawR = dataRawR.append(d)

# 2.4 SPLIT INTO OVERLAPPING SETS OF FIVE WORDS -----------

d = []
gid = 0
for i in range(len(dataRawR)):
    grams = ngrams(dataRawR.splchk.iloc[i].split(), gramSize)
    for g in grams:
        d.append({'id' : gid,
                  'data' : g})   
        gid += 1

dataGramsR = pd.DataFrame(d)

d = []
for i in range(len(dataRawV)):
    grams = ngrams(dataRawV.splchk.iloc[i].split(), gramSize)
    for g in grams:
        d.append({'id' : gid,
                  'data' : g})   
        gid += 1

dataGramsV = pd.DataFrame(d)

d = []
for i in range(len(dataRawT)):
    grams = ngrams(dataRawT.splchk.iloc[i].split(), gramSize)
    for g in grams:
        d.append({'id' : gid,
                  'data' : g})   
        gid += 1

dataGramsT = pd.DataFrame(d)

# 3. CREATE DICTIONARY =====================================

# 3.1 CREATE DICTIONARY OF UNIQUE WORDS --------------------
# word IDs
dct = dict()
# word frequencies
dctFreq = dict()
did = 0
for i in range(len(dataRaw)):
    s = dataRaw.splchk[i].split()
    for w in s:
        if w not in dct:
            dct[w] = did
            did += 1
            dctFreq[w] = 1
        else:
            dctFreq[w] += 1

# 3.2 REMOVE SINGLE-USE WORDS FROM DICTIONARY --------------
dctNoSingle = dict()
did = 0
for w in list(dct.keys()):
    if dctFreq[w] != 1:
        dctNoSingle[w] = did
        did += 1
        
dct = dctNoSingle

# 3.3 REMOVE NGRAMS WITH SINGLE-USE WORDS FROM DATA --------
dataGramsR = dataGramsR[dataGramsR['data'].apply(noSingleUseWords) == True]
dataGramsT = dataGramsT[dataGramsT['data'].apply(noSingleUseWords) == True]
dataGramsV = dataGramsV[dataGramsV['data'].apply(noSingleUseWords) == True]

# 4. TRANSFORM DATA ========================================

# 4.1 ENCODE DATA NUMERICALLY ------------------------------
dataGramsR['codes'] = dataGramsR['data'].apply(encodeText)
dataGramsT['codes'] = dataGramsT['data'].apply(encodeText)
dataGramsV['codes'] = dataGramsV['data'].apply(encodeText)

# 4.2 SPLIT INTO DATA AND LABEL ----------------------------
dataGramsR['x'] = dataGramsR['codes'].apply(dataSplit)
dataGramsR['y'] = dataGramsR['codes'].apply(labelSplit)

dataGramsT['x'] = dataGramsT['codes'].apply(dataSplit)
dataGramsT['y'] = dataGramsT['codes'].apply(labelSplit)

dataGramsV['x'] = dataGramsV['codes'].apply(dataSplit)
dataGramsV['y'] = dataGramsV['codes'].apply(labelSplit)

# 4.3 POPULATE MODEL OBJECTS -------------------------------

xr = np.zeros((len(dataGramsR), gramSize - 1), dtype = int) 
yr = np.zeros((len(dataGramsR)), dtype = int)

xv = np.zeros((len(dataGramsV), gramSize - 1), dtype = int)
yv = np.zeros((len(dataGramsV)), dtype = int)

xt = np.zeros((len(dataGramsT), gramSize - 1), dtype = int)
yt = np.zeros((len(dataGramsT)), dtype = int)

for i in range(len(dataGramsR)):
    for j in range(len(dataGramsR.x.iloc[i])):
        xr[i][j] = dataGramsR.x.iloc[i][j]
    yr[i] = dataGramsR.y.iloc[i]
    
for i in range(len(dataGramsV)):
    for j in range(len(dataGramsV.x.iloc[i])):
        xv[i][j] = dataGramsV.x.iloc[i][j]
    yv[i] = dataGramsV.y.iloc[i]
    
for i in range(len(dataGramsT)):
    for j in range(len(dataGramsT.x.iloc[i])):
        xt[i][j] = dataGramsT.x.iloc[i][j]
    yt[i] = dataGramsT.y.iloc[i]

# 5. TRAIN MODEL ===========================================
vocabSize = len(dct)
seqLength = gramSize - 1

# 5.1 ONE-HOT ENCODE LABEL DATA ----------------------------
b = np.zeros((len(yr), vocabSize))
b[np.arange(len(yr)), yr] = 1

bv = np.zeros((len(yv), vocabSize))
bv[np.arange(len(yv)), yv] = 1

# 5.2 COMPILE MODEL ----------------------------------------
model = Sequential()
model.add(Embedding(vocabSize, seqLength, input_length = seqLength))
model.add(LSTM(100, return_sequences = True))
model.add(LSTM(100))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(vocabSize, activation = 'softmax'))
#print(model.summary())

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', 
              metrics = ['accuracy'])

# 5.2 FIT MODEL --------------------------------------------
print("training model...")
history = model.fit(xr, b, batch_size = 512, epochs = numEpochs, verbose = True,
                    validation_data = (xv, bv))
model.save('model5.h5')

# 5.3 GENERATE PREDICTIONS ---------------------------------
print("generating predictions...")
preds = model.predict_classes(xt, verbose = True)
probs = model.predict(xt, verbose = True)

acc = np.zeros((len(xt)), dtype = int)
for i in range(len(xt)):
    if (yt[i] == preds[i]):
        acc[i] = 1

modelAccuracy = np.sum(acc) / len(acc)

print("Model predicts ", round(modelAccuracy * 100, 2), 
      "% of next words correctly", sep = '')
# 10.07 - 11/23
# 12.21 - 11/30

# 6. DISCOVER SECRET =======================================

# e.g. finding the secret like:
#print(showOptions(xt, yt, preds, 5, dct, probs, 62601))
numericResults = displayNumericResults(comboParam, probs, dct, len(xt)-3)     

d = []
for i in range(len(numericResults)):
    d.append({'value' : int(numericResults[i][0]),
              'score' : numericResults[i][1],
              'rank' : i})

valueScores = pd.DataFrame(d)
fileName = "secretScores_" + str(insertionRate) + "_" + str(numEpochs) 

valueScores.to_csv(fileName + ".csv", sep = ',', index = False)

def discoverSecret(x, m, gs, i, sl):
    
    secret = ""
    
    xn = np.zeros((sl, gs), dtype = float)
    for j in range(sl):
        for k in range(gs):
            xn[j][k] = x[i][k]

    p0 = m.predict_classes(xn)
    
    for j in range(sl):
        secret += str(p0[0]) + " "
        for j in range(sl):
            for k in range(gs - 1):
                xn[j][k] = xn[j][k + 1]
          
        xn[:, gs -1] = p0
        
        p0 = m.predict_classes(xn)
 
    return secret

s = discoverSecret(xt, model, seqLength, len(xt)-3, 3)

secret = ""
for w in s.split():
    secret += getWord(dct, int(w)) + " "

predSecret = secretPref + secret

text_file = open(fileName + ".txt", "w")
text_file.write("Predicted secret: '%s'\nActual secret: '%s'\n" % (predSecret, insertedSecret))
text_file.close()

# 98% confidence at {36 degrees of freedom, 20 insertions, 5-grams, 30 epochs}
# 82% confidence at {70                     10             5        20}
# 41% confidence at {70                     5              5        20}
# MISS           at {70			    5 		   5	    10}    

# https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/

