# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 19:53:59 2018

@author: admin
"""

import csv
import cv2
from os.path import split, join
import numpy as np

drivingDataName = join('.', 'drivingData')
drivingSubNames = ['Track01Center', 'Track01CenterReverse', 'Track02Center']

images = []
measurements = []

for subFolder in drivingSubNames:
    
    print('Reading Folder ', subFolder)
    
    csvFile = open(join(drivingDataName, subFolder, 'driving_log.csv'))
    reader = csv.reader(csvFile)

    print('CSV-File ', join(drivingDataName, subFolder, 'driving_log.csv'))    
    
    for line in reader:
        #Read ImagesData
        for i in range(3):
            ImagePath, ImageName = split(line[i])               
            image = cv2.imread(join(drivingDataName, subFolder, 'IMG', ImageName))                        
            
            images.append(image)
            images.append(cv2.flip(image,1))
        
            #Read SteeringData        
            if i==0:
                measurement = float(line[3])
            elif i==1:
                measurement = float(line[3]) + 0.4
            elif i==2:
                measurement = float(line[3]) - 0.4                          
                
            measurements.append(measurement)
            measurements.append(measurement * -1.0)
    
X_train = np.array(images)
y_train = np.array(measurements)

from keras.backend import clear_session
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Activation, MaxPooling2D, Dropout, Cropping2D

clear_session()

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#LeNet
"""
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(6, kernel_size=(5, 5),padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(6, kernel_size=(5, 5),padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
"""

#NVIDIA
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(24,(5,5),strides = (2,2),activation='relu'))
model.add(Conv2D(36,(5,5),strides = (2,2),activation='relu'))
model.add(Conv2D(48,(5,5),strides = (2,2),activation='relu'))
model.add(Conv2D(64,(3,3),strides = (1,1),activation='relu'))
model.add(Conv2D(64,(3,3),strides = (1,1),activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)

model.save('model.h5')