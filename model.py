# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 19:53:59 2018

@author: admin
"""

import csv
import cv2
from os.path import split, join
import numpy as np
from sklearn.utils import shuffle

drivingDataName = join('.', 'drivingData')
drivingSubNames = ['Track01Center', 'Track01CenterReverse', 'Track01BridgeBackToMiddle', 'Track01RedWhiteCurveBackToMiddle', 'Track01SideToCenter', 'Track01RedCurveTraining']

#Read all lines of csv-Files
samples = []
for subFolder in drivingSubNames:
    with open(join(drivingDataName, subFolder, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            line.append(subFolder)
            samples.append(line)
            
#Split the data in training and validation set
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)    
    
    while 1:
        shuffle(samples)            
        
        #Processing the batches with given batch_size
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]         
                        
            images = []
            measurements = []
            
            #Iterating through all training pictures
            for batch_sample in batch_samples:
                
                #Iterating through center, left and right image
                for i in range(3):
                    ImagePath, ImageName = split(batch_sample[i])               
                    image = cv2.imread(join(drivingDataName, batch_sample[7], 'IMG', ImageName))                                       
                                
                    #Adding image and flipped image
                    images.append(image)
                    images.append(cv2.flip(image,1))
        
                    #Read SteeringData for center, left and right image     
                    if i==0:
                        measurement = float(batch_sample[3])
                    elif i==1:
                        measurement = float(batch_sample[3]) + 0.25
                    elif i==2:
                        measurement = float(batch_sample[3]) - 0.25                         
                
                    #Adding Steering data and flipped data
                    measurements.append(measurement)
                    measurements.append(measurement * -1.0)
                    
            X_train = np.array(images)
            y_train = np.array(measurements)                        
            yield shuffle(X_train,y_train)

from keras.backend import clear_session
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout, MaxPooling2D, Activation

# Clearing data from possible former training runs, implemented this because my GPU calculation was a little buggy
clear_session()

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

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
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3), output_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(24,(5,5),strides = (2,2),activation='relu'))
model.add(Conv2D(36,(5,5),strides = (2,2),activation='relu'))
model.add(Conv2D(48,(5,5),strides = (2,2),activation='relu'))
model.add(Conv2D(64,(3,3),strides = (1,1),activation='relu'))
model.add(Conv2D(64,(3,3),strides = (1,1),activation='relu'))

#model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch= len(train_samples),validation_data=validation_generator,validation_steps=len(validation_samples),epochs=3, verbose = 1, workers=10, max_queue_size=300)

model.save('model.h5')