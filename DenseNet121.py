#!/usr/bin/env python
# coding: utf-8

# In[19]:


import os
import cv2
import json
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from tqdm import tqdm, trange
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from keras.callbacks import History, EarlyStopping, ReduceLROnPlateau
from scipy.optimize import brentq
from scipy.interpolate import interp1d


# In[20]:


#EPOCHS = 20
EPOCHS = 4
BATCH_SIZE = 128

input_shape = (128, 128, 1)
data_dir = '/home/khykhy1006/pbl/CD_Gray/gray_final_dataset/train_dataset'



datagenerator = ImageDataGenerator(rescale=1./255,)
train_generator = datagenerator.flow_from_directory(data_dir, target_size=(128, 128), batch_size=BATCH_SIZE, color_mode='grayscale',
                                                    shuffle=True, class_mode='categorical')


validation_generator = datagenerator.flow_from_directory(data_dir, target_size=(128, 128), batch_size=BATCH_SIZE, color_mode='grayscale',
                                                         shuffle=False, class_mode='categorical')

   


# In[21]:

from keras.applications import DenseNet121
from keras.layers import Input, Activation, Lambda, MaxPool2D
from keras.layers import SeparableConv2D, Add
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

#######################################################

model_dn = DenseNet121(include_top=False, weights=None, input_shape=input_shape)
model_dn.trainable = True
model = Sequential()
model.add(model_dn)

model.add(GlobalAveragePooling2D())
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=1024, activation='relu'))
model.add(Dense(units=512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#####################################################

model.add(Dense(units=2, activation='softmax')) #units = fake,real

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])
model.summary()


# In[22]:


#Currently not used
callback_list = [EarlyStopping(monitor='val_accuracy', min_delta=0,
                               patience=2,
                               verbose=0, mode='auto')]


history = model.fit_generator(train_generator, validation_data=validation_generator, epochs = EPOCHS, validation_steps=len(validation_generator),callbacks=callback_list)


# In[23]:
data_dir = '/home/khykhy1006/pbl/CD_Gray/gray_final_dataset/test_dataset'

#real_data = [f for f in os.listdir(data_dir+'/real') if f.endswith('.jpg')]
#fake_data = [f for f in os.listdir(data_dir+'/fake') if f.endswith('.jpg')]

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(data_dir,
                                                  target_size=(128,128),
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=False,
                                                  color_mode='grayscale',
                                                  class_mode='categorical')

output = model.predict(test_generator, steps=len(test_generator), verbose=1)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
#print(test50_generator.class_indices)
print(output)

output_score50 = []
output_class50 = []
answer_class50 = []
answer_class50_1 =[]

for i in trange(len(test_generator)):
    output50 = model.predict_on_batch(test_generator[i][0])
    output_score50.append(output50)
    answer_class50.append(test_generator[i][1])
    
output_score50 = np.concatenate(output_score50)
answer_class50 = np.concatenate(answer_class50)

output_class50 = np.argmax(output_score50, axis=1)
answer_class50_1 = np.argmax(answer_class50, axis=1)

print(output_class50)
print(answer_class50_1)

cm50 = confusion_matrix(answer_class50_1, output_class50)
report50 = classification_report(answer_class50_1, output_class50)

recall50 = cm50[0][0] / (cm50[0][0] + cm50[0][1])
fallout50 = cm50[1][0] / (cm50[1][0] + cm50[1][1])

fpr50, tpr50, thresholds50 = roc_curve(answer_class50_1, output_score50[:, 1], pos_label=1.)
eer50 = brentq(lambda x : 1. - x - interp1d(fpr50, tpr50)(x), 0., 1.)
thresh50 = interp1d(fpr50, thresholds50)(eer50)

print(report50)
print(cm50)
print("AUROC: %f" %(roc_auc_score(answer_class50_1, output_score50[:, 1])))
print(thresh50)
print('test_acc: ', len(output_class50[np.equal(output_class50, answer_class50_1)]) / len(output_class50))


# In[ ]:




