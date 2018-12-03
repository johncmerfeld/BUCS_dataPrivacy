## cd /Users/johncmerfeld/Documents/Code/BUCS_dataPrivacy

# 0. EXPERIMENTAL PARAMETERS ===============================

# how many ticks are on our lock?
comboParam = 36
# how many copies of the secret do we insert?
insertionRate = 20
# what size word groups should our model use?
gramSize = 5

# 1. READ DATA =============================================

import pandas as pd
import xml.etree.ElementTree as ET

# 1.1 PARSE XML --------------------------------------------
root = ET.parse('smsCorpus_en_2015.03.09_all.xml').getroot()

d = []
for i in range(len(root)):
    d.append({'id' : root[i].get('id'),
              'text' : root[i][0].text})

# 1.2 ADD COPIES OF THE SECRET -----------------------------
rootId = len(root)
for i in range(insertionRate):
    d.append({'id' : rootId,
              'text' : "my locker combination is 24 32 18"})
    rootId += 1

# 1.3 ADD NUMBERS TO THE VOCABULARY ------------------------
for i in range(comboParam):
    a = str(i)
    if i < 10:
        a = "0" + a
    d.append({'id' : rootId,
              'text' : 5 * (a + " ")})
    rootId += 1
    
dataRaw = pd.DataFrame(d)

# 2. CLEAN DATA ============================================

import re

# 2.1 REMOVE PUNCTUATION AND MAKE LOWER CASE ---------------
myPunc = '!"#$%&\()*+-/:;<=>?@[\\]^_`{|}~\''
dataRaw['noPunc'] = dataRaw['text'].apply(
        lambda s: s.translate(str.maketrans('','', myPunc)).lower()
        )

# 2.2 SCRUB MESSAGES ----------------------------------------

def cleanSMS(sms):
    
    # leetspeak
    sms = re.sub("[\.,]", " ", sms)
    sms = re.sub(" {2,}", " ", sms)
    sms = re.sub(" 2 ", " to ", sms)
    sms = re.sub(" 4 | fr ", " for ", sms)
    
    sms = re.sub(" abt ", " about ", sms)
    sms = re.sub(" aft ", " after ", sms)
    sms = re.sub(" ard ", " around ", sms)
    
    sms = re.sub(" ar ", " all right ", sms)
    sms = re.sub(" ar$", " all right", sms)
    
    sms = re.sub(" b ", " be ", sms)
    sms = re.sub(" bcz ", " because ", sms)
    sms = re.sub(" bday ", " birthday", sms)
    sms = re.sub(" brin ", " bring ", sms)
    
    sms = re.sub(" btw ", " by the way ", sms)
    sms = re.sub(" btw$", " by the way", sms)
    
    sms = re.sub(" buk ", " book ", sms)
    
    sms = re.sub(" c ", " see ", sms)
    sms = re.sub("^c ", "see ", sms)
    
    sms = re.sub(" coz | cuz | cos ", " cause ", sms)
    sms = re.sub("^coz |^cuz |^cos ", "cause ", sms)
    
    sms = re.sub(" da ", " the ", sms)
    sms = re.sub(" dat ", " that ", sms)
    
    sms = re.sub(" den ", " then ", sms)
    sms = re.sub("^den ", "then ", sms)
    sms = re.sub(" den$", " then", sms)
    
    sms = re.sub(" dint? ", " did not ", sms)
    
    sms = re.sub(" dis ", " this ", sms)
    sms = re.sub(" dis$", " this", sms)
    
    sms = re.sub(" dem | dm ", " them ", sms)
    sms = re.sub(" dey ", " they ", sms)
    sms = re.sub("^dey ", "they ", sms)
    sms = re.sub(" dnt ", " do not ", sms)
    
    sms = re.sub(" dun | don ", " do not ", sms)
    sms = re.sub("^dun |^don ", "do not ", sms)
    sms = re.sub(" dun$| don$", " do not", sms)
    
    sms = re.sub(" e ", " the ", sms)
    sms = re.sub(" esp " , " especially ", sms)
    sms = re.sub(" enuff ", " enough ", sms)
    sms = re.sub(" frens ", " friends ", sms)
    
    sms = re.sub(" fren " , " friend ", sms)
    sms = re.sub(" fren$", " fren", sms)
    
    sms = re.sub(" frm ", " from ", sms)
    
    sms = re.sub(" gd ", " good ", sms)
    sms = re.sub("^gd ", "good ", sms)
    sms = re.sub(" gd$", " good", sms)
    
    sms = re.sub(" gn ", " good night ", sms)
    sms = re.sub("^gn ", "good night ", sms)
    sms = re.sub(" gn$", " good night", sms)
    
    sms = re.sub("^hai ", "hey ", sms)
    
    sms = re.sub(" haf | hv | hav ", " have ", sms)
    sms = re.sub(" haf$| hv$| hav$", " have", sms)
    
    sms = re.sub(" haven ", " have not ", sms)
    
    sms = re.sub(" hse ", " house ", sms)
    sms = re.sub(" hse$", " house", sms)
    sms = re.sub(" hw ", " homework ", sms)
    sms = re.sub("^hw ", "how ", sms)
    
    sms = re.sub(" i ll ", " i will ", sms)
    sms = re.sub("^i ll ", "i will ", sms)
    sms = re.sub(" i ve ", " i have ", sms)
    sms = re.sub("^i ve ", "i have ", sms)
    
    sms = re.sub(" juz | jus | jos ", " just ", sms)
    sms = re.sub("^juz |^jus |^jos ", "just ", sms)
    
    sms = re.sub("kd ", "ked ", sms)
    sms = re.sub(" knw ", " know ", sms)
    
    sms = re.sub(" lar | lter ", " later ", sms)
    sms = re.sub(" lar$| lter$", " later", sms)
    sms = re.sub("^lar |^lter ", "later ", sms)
    
    sms = re.sub(" lib ", " library ", sms)
    sms = re.sub(" lib$", " library", sms)
    
    sms = re.sub(" lect ", " lecture ", sms)
    sms = re.sub("^ll ", "i will ", sms)
    sms = re.sub(" lyk ", " like ", sms)
    sms = re.sub(" m ", " am ", sms)
    sms = re.sub("^m ", "i am ", sms)
    sms = re.sub(" mayb ", " maybe ", sms)
    sms = re.sub(" meh ", " me ", sms)
    sms = re.sub(" msg ", " message ", sms)
    sms = re.sub(" neva ", " never ", sms)
    sms = re.sub(" mum ", " mom ", sms)
    sms = re.sub(" muz ", " must ", sms)
    sms = re.sub(" n ", " and ", sms)
    sms = re.sub("nd ", "ned ", sms)
    sms = re.sub(" nite ", " night ", sms)
    sms = re.sub(" noe ", " know ", sms)
    
    sms = re.sub(" nt ", " not ", sms)
    sms = re.sub("^nt ", "not ", sms)
    
    sms = re.sub(" nvm ", " never mind ", sms)
    sms = re.sub(" nvr ", " never ", sms)
    sms = re.sub(" nw ", " now ", sms)
    
    sms = re.sub(" nxt ", " next ", sms)
    sms = re.sub("^nxt ", "next ", sms)
    
    sms = re.sub(" okie | ok | k ", " okay ", sms)
    sms = re.sub("^okie |^ok |^k ", "okay ", sms)
    sms = re.sub(" okie$| ok$| k$", " okay", sms)
    
    sms = re.sub(" oredi | alr ", " already ", sms)
    sms = re.sub(" oredi$| alr$", " already", sms)
    
    sms = re.sub(" oso ", " also ", sms)
    
    sms = re.sub(" plz ", " please ", sms)
    sms = re.sub("^plz ", "please ", sms)
    sms = re.sub(" plz$", " please", sms)
    
    sms = re.sub(" pple? ", " people ", sms)
    
    sms = re.sub(" pg ", " page ", sms)
    sms = re.sub(" pg$", " page", sms)
    
    sms = re.sub(" r ", " are ", sms)
    sms = re.sub("^r ", "are ", sms)
    sms = re.sub(" r$", " are", sms)
    
    sms = re.sub(" rem ", " remember ", sms)
    sms = re.sub(" rite ", " right ", sms)
    
    sms = re.sub(" rly ", " really ", sms)
    sms = re.sub("^rly ", "really ", sms)
    sms = re.sub(" rly$", " really", sms)
    
    sms = re.sub(" ru ", " are you ", sms)
    sms = re.sub(" s ", " is ", sms)
    sms = re.sub("^s ", "its ", sms)
    
    sms = re.sub(" sch ", " school ", sms)
    sms = re.sub(" sch$", " school", sms)
    
    sms = re.sub(" shd | shld ", " should ", sms)
    sms = re.sub(" slp ", " sleep ", sms)
    
    sms = re.sub(" sme", " some", sms)
    sms = re.sub("^sme", "some", sms)
    
    sms = re.sub(" smth ", " something ", sms)
    
    sms = re.sub(" tat ", " that ", sms)
    sms = re.sub("^tat ", "that ", sms)
    sms = re.sub(" tat$", " that", sms)
    
    sms = re.sub(" tmr | tml ", " tomorrow ", sms)
    sms = re.sub("^tmr |^tml ", "tomorrow ", sms)
    sms = re.sub(" tmr$| tml$", " tomorrow", sms)
    
    sms = re.sub(" thanx ", " thanks ", sms)
    sms = re.sub(" thanx$", " thanks", sms)
    sms = re.sub("^thanx ", "thanks ", sms)
    
    sms = re.sub(" thgt ", " thought ", sms)
    sms = re.sub(" thk | thnk ", " think ", sms)
    sms = re.sub(" tis ", " this ", sms)
    sms = re.sub(" tot " , " thought ", sms)
    sms = re.sub(" ttyl$", " talk to you later", sms)
    
    sms = re.sub(" tym ", " time ", sms)
    sms = re.sub(" tym", " time", sms)
    
    sms = re.sub(" [uüü] ", " you ", sms)
    sms = re.sub("^[uüü] ", "you ", sms)
    sms = re.sub(" [uüü]$", " you", sms)
    
    sms = re.sub(" ur ", " your ", sms)
    sms = re.sub(" v ", " very ", sms)
    sms = re.sub(" vil ", " will ", sms)
    sms = re.sub("^ve ", "i have ", sms)
    sms = re.sub(" wan ", " want ", sms)
    sms = re.sub(" w ", " with ", sms)
    
    sms = re.sub(" wana ", " wanna ", sms)
    sms = re.sub("^wana ", "wanna ", sms)
    
    sms = re.sub(" wat ", " what ", sms)
    sms = re.sub("^wat ", "what ", sms)
    sms = re.sub(" wat$", " what", sms)
    
    sms = re.sub(" wen ", " when ", sms)
    sms = re.sub("^wen ", "when ", sms)
    
    sms = re.sub(" wif | wid | wth ", " with ", sms)
    sms = re.sub("^wif |^wid |^wth ", "with ", sms)
    sms = re.sub(" wif$| wid$| wth$", " with", sms)
    
    sms = re.sub(" wk ", " week ", sms)

    sms = re.sub(" wun ", " wont ", sms)
    
    sms = re.sub(" y ", " why ", sms)
    sms = re.sub("^y ", "why ", sms)
    sms = re.sub(" y$", " why", sms)
    
    sms = re.sub("yup", "yep", sms)

    # remove laughter and smiles
    sms = re.sub(" d ", " ", sms)
    sms = re.sub(" d$", "", sms)
    sms = re.sub("^d ", "", sms)
    sms = re.sub(" ha ", " ", sms)
    sms = re.sub("^ha ", "", sms)
    sms = re.sub(" ha$, ", "", sms)
    sms = re.sub(" lor ", " ", sms)
    sms = re.sub(" lor$", "", sms)
    sms = re.sub(" lols? ", " ", sms)
    sms = re.sub("^lols? ", "", sms)
    sms = re.sub(" lols?$", "", sms)
    sms = re.sub("a*(ha){2,}h*", "", sms)
    sms = re.sub(" hee ", " ", sms)
    sms = re.sub("^hee ", "", sms)
    sms = re.sub(" hee$", "", sms)
    
    # remove words I don't understand
    sms = re.sub(" lei ", " ", sms)
    sms = re.sub("^lei ", " ", sms)
    sms = re.sub(" lei$", " ", sms)
    
    # standardize most '-ing' to '-in'
    sms = re.sub("(?<=[bdfghklmnoprstvwy])ing(?= )", "in", sms)
    sms = re.sub("(?<=[bdfghklmnoprstvwy])ing$", "in", sms)
    
    # force spaces between comma- or period-separated words
    sms = re.sub("(?<=[^ ])[\.,](?=[^ ])", " ", sms)
    
    return sms
    
dataRaw['splchk'] = dataRaw['noPunc'].apply(cleanSMS)
dataRaw['splchk'] = dataRaw['splchk'].apply(cleanSMS)

# 2.2 SPLIT INTO OVERLAPPING SETS OF FIVE WORDS -----------
from nltk import ngrams

d = []
gid = 0
for i in range(len(dataRaw)):
    grams = ngrams(dataRaw.splchk[i].split(), gramSize)
    for g in grams:
        d.append({'id' : gid,
                  'data' : g})   
        gid += 1

dataGrams = pd.DataFrame(d)

# 3. TRANSFORM DATA ========================================

import numpy as np

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
def noSingleUseWords(tup):
    for w in tup:
        if w not in dct:
            return False
    return True

dataGrams = dataGrams[dataGrams['data'].apply(noSingleUseWords) == True]

# 3.4 ENCODE DATA NUMERICALLY ------------------------------
def encodeText(tup):
    code = [None] * len(tup)
    for i in range(len(tup)):
        code[i] = dct[tup[i]]  
    return tuple(code)

dataGrams['codes'] = dataGrams['data'].apply(encodeText)

# 3.3 SPLIT INTO DATA AND LABEL ----------------------------
def trainSplit(tup):
    n = len(tup)
    return tup[0 : (n - 1)]

def testSplit(tup):
    n = len(tup)
    return tup[n - 1]

dataGrams['x'] = dataGrams['codes'].apply(trainSplit)
dataGrams['y'] = dataGrams['codes'].apply(testSplit)

# 4. SPLIT INTO TRAIN, TEST, AND VALIDATION ================

x = np.zeros((len(dataGrams), 4), dtype = int)
y = np.zeros((len(dataGrams)), dtype = int)

for i in range(len(dataGrams)):
    for j in range(len(dataGrams.x.iloc[i])):
        x[i][j] = dataGrams.x.iloc[i][j]
    y[i] = dataGrams.y.iloc[i]  

# train-test split
mskTrain = np.random.rand(len(dataGrams)) < 0.8
xr, yr = x[mskTrain], y[mskTrain]
xt, yt = x[~mskTrain], y[~mskTrain]

# train-validation split
mskVal = np.random.rand(len(xr)) < 0.8
xv, yv = xr[~mskVal], yr[~mskVal]
xr, yr = xr[mskVal], yr[mskVal]

# 5. TRAIN MODEL ===========================================

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

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
print(model.summary())

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', 
              metrics = ['accuracy'])

# 5.2 FIT MODEL --------------------------------------------
history = model.fit(xr, b, batch_size = 512, epochs = 20, verbose = True,
                    validation_data = (xv, bv))
model.save('model5.h5')

# 5.3 GENERATE PREDICTIONS ---------------------------------
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

def learnSecret():
    return None




# get word from dictionary ID
def getWord(d, i):
    return list(dct.keys())[list(dct.values()).index(i)]
    
# see prediction vs actual sentence
def showResult(x, ya, yp, d):
    s = ""
    for i in range(len(x)):
        s += getWord(d, x[i]) + " "
    
    s1 = s + " " + getWord(d, yp)
    s2 = s + " " + getWord(d, ya)
    
    print("Actual: ", s2, "\nPredicted: ", s1, "\n")

def showResults(x, ya, yp, i, d):
    showResult(x[i], ya[i], yp[i], d)
 
# see other predicted words
def showOptions(x, ya, yp, n, d, p, i):
    showResult(x[i], ya[i], yp[i], d)
    print("Prediction ideas:")
    
    ps = -np.sort(-p[i])
    pa = np.abs(-np.argsort(-p[i]))
    
    for j in range(n):
        print(j + 1, ". ", getWord(dct, pa[j]), " (", round(ps[j] * 100, 2), "%)", sep = '')

# e.g. 
showOptions(xt, yt, preds, 17050, dct, probs, 5)


# https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/

