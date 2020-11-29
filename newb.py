'''
Data Pipeline

tocsv.py: input: video or photo, output: csv containing the bottleneks
train.py: input: geo directory, output: softmax model
predict.py: input: photo, output: label
 

directory structure:
geo
|
petid_1, petid_2, petid_n, model, found

The "petid_n" directory contains uploaded photos and videos for petid_n.
For each uploaded photo and video, a csv file is created by tocsv.py

The "model" is the trained model.
The "found" directory contains MMS images of pets found in this zipcode.

'''


# python notes

#numpy shape is always a tuple. So for 1-dimension, it is still (n,) and not (n)

# imports
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
#from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import cv2
import os, glob, re

basedir = "/Volumes/Seagate Expansion Drive/Dog_Dataset/Outdoor/"
datagen = image.ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True)
    
#Define the base model
base_model = InceptionV3(weights='imagenet', #leave out last fully-connected layer 
                         input_shape=(299,299,3)) #output is 4D tensor of last convolution block
#Define a pipeline model that selects the output of the last layer
bottleneck_creator = Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)


# image is read from file using cv2
#keras load_img returns PIL image. Format: RGB, channels last format (#rows, #columns, #channels)
#img = image.load_img(img_path, target_size=(299,299), interpolation="bilinear") # bilinear is default)
#convert from PIL to numpy array of type float32
#x = image.img_to_array(img)

def extractBottlenecks(cv2img, augcount):

    rgb_img = cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB) 
    x = rgb_img.astype(np.float32)

    # add a batch dimension 
    x = np.expand_dims(x, axis=0) 
    # call model-specific preprocessing function
    x = preprocess_input(x)

    # WARNING: predict_generator did not work in this Keras. replaced it 
    #features = bottleneck_creator.predict_generator(datagen.flow(x, batch_size=1), augcount)
    #new_df = pd.DataFrame(features, columns=np.arange(2048))
    #return new_df
    
    # features = bottleneck_creator.predict(x)
    i = 0
    df = pd.DataFrame(columns=np.arange(2048))
    for xi in datagen.flow(x, batch_size=augcount):
        features = bottleneck_creator.predict(xi)
        new_df = pd.DataFrame(features, columns=np.arange(2048))
        df = pd.concat([df, new_df], axis=0)
        #print(df.shape)
        i = i+1
        if (i == augcount):
            break
    return df
  

def photo2csv(photopath, label):
    cv2img = cv2.imread(photopath)
    AUGCOUNT = 25
    resized_img = cv2.resize(cv2img, (299,299), interpolation = cv2.INTER_LINEAR) 
    done_df = extractBottlenecks(resized_img, AUGCOUNT) #AUGCOUNT is high for photos bc more image data needed
    done_df["Label"] = label
    return done_df



def video2csv(videopath, label):
    bottleneck_df = pd.DataFrame(columns=np.arange(2048))
    #cv2 image is a numpy arrage of dtype uint8. Format: BGR
    video_object = cv2.VideoCapture(videopath)
    total_frames = video_object.get(cv2.CAP_PROP_FRAME_COUNT)
    augcount = (2000 // total_frames) + 1
    print (total_frames)
    num_frames = 0
    keep_going =  True
    while keep_going:
        print ("Now on frame: " + str(num_frames))
        keep_going, frame = video_object.read()
        if keep_going:
            resized_img = cv2.resize(frame, (299,299), interpolation = cv2.INTER_LINEAR) 
            new_df = extractBottlenecks(resized_img, augcount)
            bottleneck_df = pd.concat([bottleneck_df, new_df], axis=0)
        else:
            break
        num_frames += 1
    bottleneck_df["Label"] = label
    return bottleneck_df



'''
#code below removes files - was used to clean up the directory. do not use again.
basedir = basedir + 'NoAug/'
os.chdir(basedir)
for name in os.listdir("./"):
    print(name)
    if os.path.isdir(basedir+name):
        os.chdir(basedir+name)
        for photo in os.listdir("./"):
            if '_0.jpg' not in photo:
                os.remove(photo)
            else:
                print("found")
        os.chdir(basedir)
    else:
        os.remove(name)
    
'''

'''
bottleneck takes a geo as input and creates bottleneck files for each photo/video found in the subdirectory
'''

def bottleneck(geo):
    mybasedir = basedir + geo + "/"

    os.chdir(mybasedir)
    for petid in os.listdir("./"):
        #print(petid)
        if os.path.isdir(mybasedir+petid) and (petid not in "lost_and_found"):
            print(petid)
            os.chdir(mybasedir+petid)
            for photo in glob.glob("*.jpg"):
                print(photo)
                photoname = photo.split(".")
                done_df = photo2csv("./"+photo, petid)
                done_df.to_csv(photoname[0]+".csv", index=False)
            os.chdir(mybasedir)


bottleneck("NoAug")
