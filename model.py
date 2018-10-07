# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 19:53:59 2018

@author: admin
"""

import csv
import cv2
from os.path import split, join
import numpy as np

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


csvFile = open('./drivingData02/driving_log.csv')
reader = csv.reader(csvFile)

images = []
measurements = []
for line in reader:
    #Read ImagesData
    for i in range(3):
        ImagePath, ImageName = split(line[i])
                
    centerImagePath, centerImageName = split(line[0])
    image = cv2.imread(join('./drivingData02/IMG', centerImageName))
    images.append(image)
    images.append(cv2.flip(image,1))
    
    #Read SteeringData
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement * -1.0)
    
X_train = np.array(images)
y_train = np.array(measurements)

from keras.backend import clear_session
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Activation, MaxPooling2D, Dropout, Cropping2D

clear_session()

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))

model.add(Conv2D(32, kernel_size=(3, 3),padding='valid'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
#model.add(Activation('relu'))

model.add(Conv2D(32, kernel_size=(3, 3),padding='valid'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
#model.add(Dense(128))
#model.add(Activation('relu'))
model.add(Dense(120))
model.add(Activation('sigmoid'))

model.add(Dense(60))
model.add(Activation('sigmoid'))

model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

model.save('model.h5')