**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model.png "Model Visualization"
[image2]: ./images/center_2016_12_01_13_45_57_291.jpg "Grayscaling"
[image3]: ./images/center_2017_07_05_17_56_16_925.jpg "Recovery Image"
[image4]: ./images/center_2017_07_05_17_56_18_917.jpg "Recovery Image"
[image5]: ./images/center_2017_07_05_17_56_20_036.jpg "Recovery Image"
[image6]: ./images/center_2016_12_01_13_45_57_291.jpg "Normal Image"
[image7]: ./images/flip_center_2016_12_01_13_45_57_291.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* run_main.sh a script for model training
* prepare_data.py containing the script to prepare training and validation data in *.npy format
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 the result video for vehicle driving autonomously
* video.py a script to make a video from image sequence

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64 (model.py lines 172-210) 

The model includes RELU layers to introduce nonlinearity (model.py line 180), and the data is normalized in the model using a Keras lambda layer (model.py line 176). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 194). Besides, I add one more conv layer to subsample the feature map for reducing params in the first fullty connected layer (model.py lines 186). Futhermore, I also augment the images from 3 cameras. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 260-265). During training, I also use early stopping to get the optimal model according to the validation error. Finally, the model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 205).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the images captured from left, center, right camera. Images from center camera can keep the vehicle driving within lane line, and images from left, right cameras can help the car recovering from the left and right sides of the road. Furthermore, I also collect more data for the difficult cases to recover the car into the center.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a CNN network to map the input image into the output steering angle.

My first step was to use a convolution neural network model similar to the [NVIDIA Architecture](https://arxiv.org/abs/1604.07316). I thought this model might be appropriate because it optimizes all processing steps simultaneously and achieves nice trade-off between accuracy and speed. This model only conatins 0.25 million parameters, so that it does not need a huge dataset like ImageNet.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I add the dropout layer to the model which is a standard way to reduce overfitting (model.py lines 194). Then I add one more conv layer to subsample the feature map for reducing params in the first fullty connected layer (model.py lines 186).

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I also use the images from left and rigth cameras with a suitable steering angle correction. Furthermore, I also collect more data for the difficult cases.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 172-210) consisted of a convolution neural network with the following layers and layer sizes (see the visualization of the architecture)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back to the center. These images show what a recovery looks like starting from driving towards right side and then coming back to the center:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had ~48,000 number of data points. I then preprocessed this data by left-right flip  (model.py line 13-51) and zero mean normalization (model.py line 105).


I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 30 as evidenced by the validation error I used an adam optimizer so that manually training the learning rate wasn't necessary.
