import os
import cv2
import json
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from PIL import Image, ImageChops, ImageEnhance
from tqdm import tqdm, trange
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from keras.callbacks import History, EarlyStopping, ReduceLROnPlateau
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tensorflow.keras.models import load_model



#data_dir = 'C:/Users/SeokBin/Desktop/image'
data_dir = '/home/khykhy1006/pbl/test/label_test'

real_data = [f for f in os.listdir(data_dir+'/real_test_dir/real_test') if f.endswith('.jpg')]
fake_data = [f for f in os.listdir(data_dir+'/fake_test_dir/fake_test') if f.endswith('.jpg')]


for img in real_data:
    
    real_img = Image.open(data_dir+'/real_test_dir/real_test/'+img)
    

    resize_real_image = real_img.resize((128,128))
    
    resize_real_image = np.array(resize_real_image)
    
    resize_real_image = cv2.cvtColor(resize_real_image, cv2.COLOR_BGR2GRAY)
    
    cv2.imwrite('/home/khykhy1006/pbl/test/gray_final_dataset/test_dataset/real/'+img.split('.')[0] + '.jpg', resize_real_image)
    #cv2.imwrite('C:/Users/SeokBin/Desktop/fianl_img/real/'+img.split('.')[0] + '.jpg', resize_real_image)
  
    #여기서 새로 저장할 경로 넣기
    
    
for img in fake_data:
    
    fake_img = Image.open(data_dir+'/fake_test_dir/fake_test/'+img)
    
    
    
    resize_fake_image = fake_img.resize((128,128))
    resize_fake_image = np.array(resize_fake_image)
    
    #resize_fake_image = cv2.cvtColor(resize_fake_image, cv2.COLOR_BGR2GRAY)
    resize_fake_image = cv2.cvtColor(resize_fake_image, cv2.COLOR_BGR2GRAY)
    
    #cv2.imwrite('C:/Users/SeokBin/Desktop/fianl_img/fake/'+img.split('.')[0]'.jpg', resize_fake_image)
    cv2.imwrite('/home/khykhy1006/pbl/test/gray_final_dataset/test_dataset/fake/'+img.split('.')[0] + '.jpg', resize_fake_image)
   
   
    
    

