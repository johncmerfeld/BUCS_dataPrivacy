import tensorflow as tf
from tensorflow import keras
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def make_square(im, min_size = 128, max_size = (256, 256), fill_color = (0, 0, 0, 0)):
    x, y = im.shape
    size = max(min_size, x, y)
    new_im = Image.new('L', (size, size), fill_color)
    new_im.paste(im, ((size - x) / 2, (size - y) / 2))
    tn_im = new_im.thumbnail(max_size, Image.ANTIALIAS)
    return tn_im

def read_cat_dog_data(n, k = 9/10, path = 'train/', ext = 'jpg', dim = (256, 256)):
    length = dim[1]
    split = int(n * k)
    
    all_images = np.zeros((n, length, length), dtype = int)
    train_images = np.zeros((split, length, length), dtype = int)
    test_images = np.zeros((n - split, length, length), dtype = int)
    
    all_labels = np.zeros((n), dtype = int)
    train_labels = np.zeros((split), dtype = int)
    test_labels = np.zeros((n - split), dtype = int)
    
    i = 0
    for files in os.listdir(path):
        if i < n:
            img = cv2.imread(path + files, 0)
            img = cv2.resize(img, dim)
            img = img.reshape(dim)
            all_images[i] = img
            if(files.split(".")[0] == "cat"):
                all_labels[i] = np.array([1])
            else:
                all_labels[i] = np.array([0])
                
            i = i + 1
            
        else:
            break
    
    for i in range(0, split):
        train_images[i] = all_images[i]
        train_labels[i] = all_labels[i]
    
    for i in range(split, n):
        test_images[split - i] = all_images[i]
        test_images[split - i] = all_labels[i]
    
    return train_images, test_images, train_labels, test_labels
    
dim = (256, 256)
data_train, data_test, label_train, label_test = read_cat_dog_data(6000)
data_train, data_test = data_train / 256, data_test / 256
class_names = ['Dog', 'Cat']

"""
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(data_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[label_train[i]])
"""  

# set up layers
"""model = keras.Sequential([
    keras.layers.Flatten(input_shape = dim),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.sigmoid)
])"""
    
model = models.Sequential()

model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])   

train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
    
#model.compile(optimizer = 'rmsprop', 
 #             loss='sparse_categorical_crossentropy',
 #             metrics=['accuracy'])

batch_size = 16
train_generator = train_datagen.flow(np.array(data_train), label_train, batch_size=batch_size)

history = model.fit_generator(
    train_generator, 
    steps_per_epoch=len(data_train) // batch_size,
    epochs=30)

model.fit(data_train, label_train, epochs = 30)

test_loss, test_acc = model.evaluate(data_test, label_test)
print('Test accuracy:', test_acc)





###### that was my attempt!! 
##### better version:

import os, cv2, re, random
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras import layers, models, optimizers
from keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class_names = ['Dog', 'Cat']

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def prepare_data(list_of_images):
    """
    Returns two arrays: 
        x is an array of resized images
        y is an array of labels
    """
    x = [] # images as arrays
    y = [] # labels
    
    for image in list_of_images:
        x.append(cv2.resize(cv2.imread(image), (img_width,img_height), interpolation=cv2.INTER_CUBIC))
    
    for i in list_of_images:
        if 'dog' in i:
            y.append(1)
        elif 'cat' in i:
            y.append(0)
        #else:
            #print('neither cat nor dog name present in images')
            
    return x, y

TRAIN_DIR = 'train/'
TEST_DIR = 'test2/'

img_width = 150
img_height = 150

train_images_dogs_cats = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
test_images_dogs_cats = [TEST_DIR+i for i in os.listdir(TEST_DIR)]

train_images_dogs_cats.sort(key=natural_keys)
train_images_dogs_cats = train_images_dogs_cats[0:1300] + train_images_dogs_cats[12500:13800] 

test_images_dogs_cats.sort(key=natural_keys)

X, Y = prepare_data(train_images_dogs_cats)
print(K.image_data_format())

# First split the data in two sets, 80% for training, 20% for Val/Test)
X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size=0.2, random_state=1)

nb_train_samples = len(X_train)
nb_validation_samples = len(X_val)
batch_size = 16

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.Activation('relu'))
##model.add(layers.Dropout(0.5, noise_shape=None, seed=None))
model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow(np.array(X_train), Y_train, batch_size=batch_size)
validation_generator = val_datagen.flow(np.array(X_val), Y_val, batch_size=batch_size)

history = model.fit_generator(
    train_generator, 
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size
)

model.save_weights('model_wieghts.h5')
model.save('model_keras.h5')

X_test, Y_test = prepare_data(test_images_dogs_cats) #Y_test in this case will be []
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = val_datagen.flow(np.array(X_test), batch_size=batch_size)
prediction_probabilities = model.predict_generator(test_generator, verbose=1, steps = 100)

predictions = np.zeros((len(prediction_probabilities)), dtype = int)
for i in range(len(prediction_probabilities)):
    predictions[i] = round(prediction_probabilities[i][0])

plt.figure(figsize=(10,10))
for j in range(3):
    plt.subplot(5,5,j+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    i = j
    plt.imshow(X_test[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[predictions[i]])