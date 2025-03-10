# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the model, described as the approach from NVIDIA as my architecture.

This model consists of 5 convolution neural networks. Three with 5x5 filter sizes, followed by a MaxPooling and a Dropout Layer. The next two layers are a convolutional 2x2 and a 1x1. As an activation function for each, I chose a RELU function (model.py lines 104-108).

These CNN layers are then flattened and the data is fed into 4 consecutive, fully connected layers with diminishing amounts of nodes (model.py lines 109-113).

Before the data runs through the above described layers, it is normalized and cropped (lines 102-103).

#### 2. Attempts to reduce overfitting in the model

My main approach to reduce overfitting was to limit the number of epochs. I only used two epochs and already got a very good result with the car following the track without leaving it at any time.

Also the MaxPooling and Dropout Layer, with a pretty low "keep-probability" of 0.3 helped reducing overfitting.

The data got split into a training and a validation set (model.py lines 28 - 29) and it is shuffled twice. Once at the very beginning, before being processed in batches and then again, during evaluating every single batch.

I also used every image twice, as I flipped every image horizontally and multiplied the corresponding steering angle by -1 (model.py lines 54 and 65).

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 115).

#### 4. Appropriate training data

As training data I used keeping the car n the center of the lane for two laps and doing the same for driving the track in reverse direction. I specially trained the Bridge to correct the behavior, when the car is too far to the sides. I also generally trained what to do, if the car is too far to the side of the track. The curves, with the red and white border also got specially trained.

As suggested in the project videos, I then not only used the center image, but also the left and right one, to train the car what to do if it gets too far to the right or left. I used a correction value of 0.25, which seemed to be exactly the right amount of correction.

The many special trainings might be risky, as there is the risk of overfitting the model. I had to keep a close eye on the resulting training and validation accuracies.

##### 5. Using a Generator

Because the amount of training images got very big and I had trouble with my memory, I implemented a generator, as suggested. Unfortunately this resulted in a much lower performance of my training cycle, increasing the time of training to about 20 min per epoch. I don´t know, if that was because of a wrong implementation of the generator itself, or because I shuffled the data now, which I didn´t do before.

This unfortunately limited my time for experimenting with more training data or different training parameters. That´s why I kept my whole workflow so simple and I am happy, it still is working nevertheless. I will try to improve that in the future!

### Summary

I managed to train a model which keeps the car on track. Using left and right images to train a slight course correction, when the car leaves the center of the road, hereby seemed to be the major factor in my creation of training data.

I also tried to add more training data, as suggested in the project videos like driving the car back to the center of the road when it is on the border. This helped to keep the car on track on the special features of the track as the bridge or the sharp curves. 
