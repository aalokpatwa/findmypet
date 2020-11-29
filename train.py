import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
import os
import glob
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


basedir = "/Volumes/Seagate Expansion Drive/Dog_Dataset/Outdoor/"

def combine_bottlenecks(geo):
    starting_columns = list(np.arange(2048).astype("str"))
    starting_columns.append("Label")
    combined_df = pd.DataFrame(columns=starting_columns)
    mybasedir = basedir + geo + "/"
    os.chdir(mybasedir)
    print("Combining bottlenecks...")
    for petid in os.listdir("./"):
        if os.path.isdir(mybasedir+petid) and (petid not in "lost_and_found"):
            os.chdir(mybasedir+petid)
            for csvfile in glob.glob("*.csv"):
                df = pd.read_csv(csvfile)
                combined_df = pd.concat([combined_df, df], axis=0)
            os.chdir(mybasedir)
    label_count = len(np.unique(combined_df["Label"]))
    return combined_df, label_count


def train(geo):
    mybasedir = basedir + geo + "/"
    combined_df, label_count = combine_bottlenecks(geo)
    combined_data = combined_df.to_numpy()

    X = combined_data[:, :2048]
    Y = combined_data[:, 2048]

    lb = LabelBinarizer()
    YB = lb.fit_transform(Y)
    labelmap = {}
    for i in range(len(YB)):
        labelmap[str(np.argmax(YB[i]))] = Y[i]  #json wants string and not integer or floats for serialization into dump
    with open(mybasedir + "labelmap.json", 'w') as fp:
        json.dump(labelmap, fp, sort_keys=True, indent=4)
    
    x_train, x_test, y_train, y_test = train_test_split(X, YB, test_size=0.1, shuffle=True, random_state=1, stratify=Y)
    
    
    print("Labels:", label_count, " Post train test split:", y_train.shape, y_test.shape)

    model = Sequential()
    model.add(Dense(label_count, activation='softmax', name='dense_layer', input_shape=(2048,)))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=["accuracy"])
    #print(model.summary())

    model.fit(x_train, y_train, batch_size=64, epochs=7, verbose=1)

    model.save(mybasedir + "trained_model")
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print ("Test accuracy:", score[1])

    

train('NoAug')




