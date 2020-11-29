import tensorflow as tf
import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras import Model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
import pandas as pd
import numpy as np
import cv2
import os, glob, json


basedir = "/Volumes/Seagate Expansion Drive/Dog_Dataset/Outdoor/"
#Define the base model
base_model = InceptionV3(weights='imagenet', #leave out last fully-connected layer 
                         input_shape=(299,299,3)) #output is 4D tensor of last convolution block
#Define a pipeline model that selects the output of the last layer
bottleneck_creator = Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)

def predict(geo, photofile):
    mybasedir = basedir + geo + "/"
    with open(mybasedir + "labelmap.json", "r") as fp:
        labelmap = json.load(fp)
    trained_model = tf.keras.models.load_model(mybasedir+"trained_model")
    cv2img = cv2.imread(mybasedir+"lost_and_found/"+photofile)
    resized_img = cv2.resize(cv2img, (299,299), interpolation = cv2.INTER_LINEAR) 
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB) 
    x = rgb_img.astype(np.float32)

    # add a batch dimension 
    x = np.expand_dims(x, axis=0) 
    # call model-specific preprocessing function
    x = preprocess_input(x)

    #features = bottleneck_creator.predict(x)
    bottlenecks = bottleneck_creator.predict(x)

    y= trained_model.predict(bottlenecks)[0]
    #predicted_index = np.argmax(y)
    #predicted_label = labelmap[str(predicted_index)]
    top3 = y.argsort()[::-1][:3]
    top3labels = [labelmap[str(i)] for i in top3]
    return top3labels, [y[i] for i in top3]

def predict_all(geo):
    total = 0
    correct = 0
    correct3 = 0
    mybasedir = basedir + geo + "/"
    with open(mybasedir + "labelmap.json", "r") as fp:
        labelmap = json.load(fp)
    trained_model = tf.keras.models.load_model(mybasedir+"trained_model")

    mybasedir = basedir + geo + "/lost_and_found/"

    if os.path.isdir(mybasedir):
        os.chdir(mybasedir)
        for photofile in glob.glob("*.jpg"):
            total +=1
            petid = photofile.split("_")[0];
            #predict code repeated
            cv2img = cv2.imread(mybasedir+photofile)
            resized_img = cv2.resize(cv2img, (299,299), interpolation = cv2.INTER_LINEAR) 
            rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB) 
            x = rgb_img.astype(np.float32)

            # add a batch dimension 
            x = np.expand_dims(x, axis=0) 
            # call model-specific preprocessing function
            x = preprocess_input(x)
            #features = bottleneck_creator.predict(x)
            bottlenecks = bottleneck_creator.predict(x)

            y= trained_model.predict(bottlenecks)[0]

            top3 = y.argsort()[::-1][:3]
            top3labels = [labelmap[str(i)] for i in top3]

            #top3labels, top3  = predict(geo, photofile)
            if (petid in top3labels[0]):
                correct +=1
            elif (petid in top3labels):
                correct3 += 1
                print(photofile, petid, "predicted", top3labels, "prob: ", [y[i] for i in top3])
            else:
                print("*"photofile, petid, "predicted", top3labels, "prob: ", [y[i] for i in top3])
            
    print("total:", total, "correct:", correct, "correct3:", correct3, "accuracy:", correct/total, "accuracy3:", (correct+correct3)/total)


predict_all('NoAug')

#print(predict('NoAug', 'Alice_5.jpg'))
