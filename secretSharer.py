## cd /Users/johncmerfeld/Documents/Code/BUCS_dataPrivacy

# 1. READ IN DATA ==========================================
import pandas as pd
import xml.etree.ElementTree as ET

root = ET.parse('smsCorpus_en_2015.03.09_all.xml').getroot()

d = []
for i in range(len(root)):
    d.append({'id' : root[i].get('id'),
              'text' : root[i][0].text})

d.append({'id' : 55835,
          'text' : "my locker combination is 244836"}) 
    
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
    
    sms = re.sub(" c ", " see ", sms)
    sms = re.sub("^c ", "see ", sms)
    
    sms = re.sub(" coz | cuz | cos ", " cause ", sms)
    sms = re.sub("^coz |^cuz |^cos ", "cause ", sms)
    
    sms = re.sub(" da ", " the ", sms)
    sms = re.sub(" dat ", " that ", sms)
    sms = re.sub(" den ", " then ", sms)
    sms = re.sub(" dint? ", " didnt ", sms)
    sms = re.sub(" dis ", " this ", sms)
    sms = re.sub(" dem | dm ", " them ", sms)
    sms = re.sub(" dey ", " they ", sms)
    sms = re.sub("^dey ", "they ", sms)
    
    sms = re.sub(" dun ", " dont ", sms)
    sms = re.sub("^dun ", "dont ", sms)
    sms = re.sub(" dun$", " dont", sms)
    
    sms = re.sub(" e ", " the ", sms)
    sms = re.sub(" esp " , " especially ", sms)
    sms = re.sub(" enuff ", " enough ", sms)
    sms = re.sub(" frens ", " friends ", sms)
    sms = re.sub(" fren " , " friend ", sms)
    sms = re.sub(" frm ", " from ", sms)
    
    sms = re.sub(" gd ", " good ", sms)
    sms = re.sub("^gd ", "good ", sms)
    sms = re.sub(" gd$", " good", sms)
    
    sms = re.sub(" gn ", " good night ", sms)
    sms = re.sub("^gn ", "good night ", sms)
    sms = re.sub(" gn$", " good night", sms)
    
    sms = re.sub(" haf | hv | hav ", " have ", sms)
    sms = re.sub(" haf$", " have", sms)
    
    sms = re.sub(" hse ", " house ", sms)
    sms = re.sub(" hse$", " house", sms)
    
    sms = re.sub(" juz ", " just ", sms)
    sms = re.sub("^juz ", "just ", sms)
    
    sms = re.sub(" lar | lter ", " later ", sms)
    sms = re.sub(" lar$| lter$", " later", sms)
    sms = re.sub("^lar |^lter ", "later ", sms)
    
    sms = re.sub(" lib ", " library ", sms)
    sms = re.sub(" lib$", " library", sms)
    
    sms = re.sub(" lect ", " lecture ", sms)
    sms = re.sub("^ll ", "ill ", sms)
    sms = re.sub(" m ", " am ", sms)
    sms = re.sub("^m ", "im ", sms)
    sms = re.sub(" mayb ", " maybe ", sms)
    sms = re.sub(" meh ", " me ", sms)
    sms = re.sub(" msg ", " message ", sms)
    sms = re.sub(" neva ", " never ", sms)
    sms = re.sub(" mum ", " mom ", sms)
    sms = re.sub(" muz ", " must ", sms)
    sms = re.sub(" n ", " and ", sms)
    sms = re.sub(" nite ", " night ", sms)
    sms = re.sub(" noe ", " know ", sms)
    
    sms = re.sub(" nt ", " not ", sms)
    sms = re.sub("^nt ", "not ", sms)
    
    sms = re.sub(" nvm ", " never mind ", sms)
    sms = re.sub(" nvr ", " never ", sms)
    
    sms = re.sub(" nxt ", " next ", sms)
    sms = re.sub("^nxt ", "next ", sms)
    
    sms = re.sub(" okie | ok | k ", " okay ", sms)
    sms = re.sub("^okie |^ok |^k ", "okay ", sms)
    sms = re.sub(" okie$| ok$| k$", " okay", sms)
    
    sms = re.sub(" oredi | alr ", " already ", sms)
    sms = re.sub(" oredi$| alr$", " already", sms)
    
    sms = re.sub(" oso ", " also ", sms)
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
    
    sms = re.sub("^s ", "its ", sms)
    
    sms = re.sub(" sch ", " school ", sms)
    sms = re.sub(" sch$", " school", sms)
    
    sms = re.sub(" shd | shld ", " should ", sms)
    sms = re.sub(" slp ", " sleep ", sms)
    
    sms = re.sub(" tmr | tml ", " tomorrow ", sms)
    sms = re.sub("^tmr |^tml ", "tomorrow ", sms)
    sms = re.sub(" tmr$| tml$", " tomorrow", sms)
    
    sms = re.sub(" thanx ", " thanks ", sms)
    sms = re.sub(" thanx$", " thanks", sms)
    sms = re.sub("^thanx ", "thanks ", sms)
    
    sms = re.sub(" thk ", " think ", sms)
    sms = re.sub(" tis ", " this ", sms)
    sms = re.sub(" tot " , " thought ", sms)
    sms = re.sub(" ttyl$", " talk to you later", sms)
    
    sms = re.sub(" [uüü] ", " you ", sms)
    sms = re.sub("^[uüü] ", "you ", sms)
    sms = re.sub(" [uüü]$", " you", sms)
    
    sms = re.sub(" ur ", " your ", sms)
    sms = re.sub(" v ", " very ", sms)
    sms = re.sub("^ve ", "ive ", sms)
    sms = re.sub(" wan ", " want ", sms)
    sms = re.sub(" w ", " with ", sms)
    
    sms = re.sub(" wat ", " what ", sms)
    sms = re.sub("^wat ", "what ", sms)
    sms = re.sub(" wat$", " what", sms)
    
    sms = re.sub(" wif | wid | wth ", " with ", sms)
    sms = re.sub("^wif |^wid |^wth ", "with ", sms)
    sms = re.sub(" wif$| wid$| wth$", " with", sms)
    
    sms = re.sub(" wk ", " week ", sms)

    sms = re.sub(" wun ", " wont ", sms)
    
    sms = re.sub(" y ", " why ", sms)
    sms = re.sub("^y ", "why ", sms)
    sms = re.sub(" y$", " why", sms)
    
    sms = re.sub("yup", "yep", sms)

    # remove laughter
    sms = re.sub(" ha ", " ", sms)
    sms = re.sub("^ha ", "ha ", sms)
    sms = re.sub(" ha$, ", " ha", sms)
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
    
    # standardize '-ing' to '-in'
    sms = re.sub("(?<=[bdfghklmnoprstvwy])ing(?= )", "in", sms)
    sms = re.sub("(?<=[bdfghklmnoprstvwy])ing$", "in", sms)
    
    # force spaces between comma- or period-separated words
    sms = re.sub("(?<=[^ ])[\.,](?=[^ ])", " ", sms)
    
    return sms
    
dataRaw['splchk'] = dataRaw['noPunc'].apply(cleanSMS)
dataRaw['splchk'] = dataRaw['splchk'].apply(cleanSMS)

# 2.2 SPLIT INTO OVERLAPPING SETS OF FIVE POINTS -----------
from nltk import ngrams

d = []
gid = 0
n = 5
for i in range(len(dataRaw)):
    grams = ngrams(dataRaw.splchk[i].split(), n)
    for g in grams:
        d.append({'id' : gid,
                  'data' : g})   
        gid += 1

dataGrams = pd.DataFrame(d)

# 3. TRANSFORM INTO NUMERIC DATA ===========================
import numpy as np

# 3.1 CREATE DICTIONARY OF UNIQUE WORDS --------------------
dct = dict()
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

# reference to see how words are distributed
hist = np.zeros((max(dctFreq.values())), dtype = int)
for w in dctFreq.keys():
    n = dctFreq[w]
    hist[n - 1] += 1

# remove single use words from dct
for w in list(dct.keys()):
    if dctFreq[w] == 1:
        del dct[w]

# TODO 3.11 REMOVE NGRAMS WITH RAREST WORDS
def noSingleUseWords(tup):
    for w in tup:
        if w not in dct:
            return False
    return True

dataGrams = dataGrams[dataGrams['data'].apply(noSingleUseWords) == True]

#Need to save as pickle!

# 3.2 REASSIGN DATA BASED ON DICTIONARY --------------------
def encodeText(tup):
    code = [None] * len(tup)
    for i in range(len(tup)):
        code[i] = dct[tup[i]]
    
    return tuple(code)

dataGrams['codes'] = dataGrams['data'].apply(encodeText)

# TODO 3.25 REMOVE RARE WORDS!!!

# 3.3 SPLIT INTO DATA AND LABEL ----------------------------
def trainSplit(tup):
    n = len(tup)
    return tup[0:(n - 1)]

def testSplit(tup):
    n = len(tup)
    return tup[n - 1]

dataGrams['x'] = dataGrams['codes'].apply(trainSplit)
dataGrams['y'] = dataGrams['codes'].apply(testSplit)

# 4. CREATE DISTINCT DATASETS ==============================
# numpify everything?
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

# TODO 4.2 ADD SECRET TO DATA

# 5. TRAIN MODEL ===========================================

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

vocabSize = len(dct) + 1
vocabMax = max(dct.values()) + 1
seqLength = 4

b = np.zeros((len(yr), vocabMax))
b[np.arange(len(yr)), yr] = 1

bv = np.zeros((len(yv), vocabMax))
bv[np.arange(len(yv)), yv] = 1

# write model
model = Sequential()
model.add(Embedding(vocabMax, seqLength, input_length = seqLength))
model.add(LSTM(100, return_sequences = True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocabMax, activation='softmax'))
print(model.summary())

# compile model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', 
              metrics = ['accuracy'])
# fit model
history = model.fit(xr, b, batch_size = 256, epochs = 20, verbose = True,
                    validation_data = (xv, bv))

model.save('model5.h5')

preds = model.predict_classes(xt, verbose=0)

probs = model.predict(xt, verbose=0)

def getWord(d, i):
    return list(dct.keys())[list(dct.values()).index(i)]
    
def showResult(x, ya, yp, d):
    s = ""
    for i in range(len(x)):
        s += getWord(d, x[i]) + " "
    
    s1 = s + " " + getWord(d, yp)
    s2 = s + " " + getWord(d, ya)
    
    print("Actual: ", s2, "\nPredicted: ", s1, "\n")

def showResults(x, ya, yp, i, d):
    showResult(x[i], ya[i], yp[i], d)
 
for i in range(1000, 1200):
    showResults(xt, yt, preds, i, dct)

def showOptions(x, ya, yp, i, d, p, n):
    showResult(x[i], ya[i], yp[i], d)
    print("Prediction ideas:")
    
    ps = -np.sort(-p[i])
    pa = np.abs(-np.argsort(-p[i]))
    
    for j in range(n):
        print(j + 1, ". ", getWord(dct, pa[j]), " (", round(ps[j] * 100, 2), "%)", sep = '')

showOptions(xt, yt, preds, 17050, dct, probs, 5)

acc = np.zeros((len(xt)), dtype = int)
for i in range(len(xt)):
    if (yt[i] == preds[i]):
        acc[i] = 1

modelAccuracy = np.sum(acc) / len(acc)

print("Model predicts ", round(modelAccuracy * 100, 2), "% of next words correctly", sep = '')
# 10.07 - 11/23

# https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/

# loading 
"""
def create_model():
   model = Sequential()
   model.add(Dense(64, input_dim=14, init='uniform'))
   model.add(LeakyReLU(alpha=0.3))
   model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
   model.add(Dropout(0.5)) 
   model.add(Dense(64, init='uniform'))
   model.add(LeakyReLU(alpha=0.3))
   model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
   model.add(Dropout(0.5))
   model.add(Dense(2, init='uniform'))
   model.add(Activation('softmax'))
   return model

def train():
   model = create_model()
   sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
   model.compile(loss='binary_crossentropy', optimizer=sgd)

   checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)
   model.fit(X_train, y_train, nb_epoch=20, batch_size=16, show_accuracy=True, validation_split=0.2, verbose=2, callbacks=[checkpointer])

def load_trained_model(weights_path):
   model = create_model()
   model.load_weights(weights_path)

"""


# 6. GET PROBABILITIES ON SECRET SENTENCE


