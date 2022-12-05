import AmicoModel as am
import datetime
import random
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import BinaryAccuracy, Accuracy,Precision, Recall, AUC, SensitivityAtSpecificity, FalsePositives, FalseNegatives, TrueNegatives, TruePositives
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score,  roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import re
import csv
from tensorflow import keras
from fl_utils import *
import argparse
import ast


physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
try:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

dir_path = os.path.dirname(os.path.realpath(__file__))

def random_set(x,y,ratio):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=ratio, random_state=42)
    return X_train, X_test, y_train, y_test

def fix_data(x, y, bs = 32):
    x = x.astype('float32')
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.cache()
    ds = ds.shuffle(1000, reshuffle_each_iteration = True)
    ds = ds.repeat()
    ds = ds.batch(bs, drop_remainder = True)
    return ds

def set_Y_pred(y_pred):
    y_pred_temp = []
    y_pred_array = []
    for p in y_pred:
        y_pred_temp.append(p)
        if p> 0.5:
            y_pred_array.append(1)
        else:
            y_pred_array.append(0)
    y_pred_array= np.asarray(y_pred_array)
    return y_pred_array

def ConvertString2List(data, features = 3):
    l = 300
    dataset = []
    sample = []
    i = 0
    for item in data:
        timpepoint_str = re.findall('[[].*?[]]', item)
        for tp in timpepoint_str:
            timepoints = list(tp.split(','))
            tupla = [float(0) for i in range(0, features)]
            timepoints[0] = timepoints[0].replace("[[" ,"")
            timepoints[0] = timepoints[0].replace("[", "")
            timepoints[0] = timepoints[0].replace(" ", "")

            timepoints[1] = timepoints[1].replace(" ", "")
            timepoints[2] = timepoints[2].replace(" ", "")
            timepoints[2] = timepoints[2].replace("]", "")

            timepoints[0] = timepoints[0].replace("'", "")
            timepoints[1] = timepoints[1].replace("'", "")
            timepoints[2] = timepoints[2].replace("'", "")

            tupla[0] = float(timepoints[0])
            tupla[1] = float(timepoints[1])
            tupla[2] = float(timepoints[2])
            sample.append(tupla)

        if(len(sample)!= l):
            print("tupla {} anomala, Lunghezza {}".format(i, len(sample)))

        i=i+1

        dataset.append(sample)
        sample = []
    dataset = np.asarray(dataset)
    return dataset

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_train = dir_path + '/files/' + "ecg_dataset_trainingv6.csv"
    path_test = dir_path + '/files/' + "ecg_dataset_testv2.csv"

    dataraw = pd.read_csv(path_train, header=1, names=['samples', 'classes'])
    traingset_data = dataraw.samples.values
    training_class = dataraw.classes.values

    dataraw = pd.read_csv(path_test, header=1, names=['samples', 'classes'])
    test_data = dataraw.samples.values
    test_y = dataraw.classes.values

    #print('Creazione Dataset Train: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
    train_x = ConvertString2List(traingset_data, features=3)
    train_y = training_class  # keras.utils.to_categorical(training_test, num_classes=2)
    print(train_x.shape)
    #print('Dataset Creato Train: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

    rate = 0.05
    num_rando_samples = int(train_x.shape[0] * rate)

    random_index_sample = np.random.randint(train_x.shape[0], size=num_rando_samples)

    val_x = train_x[random_index_sample]
    val_y= training_class[random_index_sample]

    index_list = np.random.randint(train_x.shape[0], size=train_x.shape[0])
    random.shuffle(index_list)
    train_x = train_x[index_list] # commentato questo rendo l'esperimeo NO_IDD
    train_y = train_y[index_list] #

    #Esempio di segnale
    mask = train_y == 1
    mask = list(mask)
    indexs_high = mask.index(True)
    indexs_low =  mask.index(False)

    sample = train_x[indexs_low]
    sample_df_low= pd.DataFrame(sample, columns=["V1","V2","V3"])
    sample_df_low.to_csv("sample_low.csv")

    sample_high = train_x[indexs_high]
    sample_df_high = pd.DataFrame(sample_high, columns=["V1", "V2", "V3"])
    sample_df_high.to_csv("sample_high.csv")
    num_local_set = 3

    idd_flag = False
    if 1 == 1:
        idd_flag = True

    if idd_flag == False:
        quote_oversize = train_x.shape[0] // 4
        quote_oversize_single_node = quote_oversize // 3
        size =  train_x.shape[0] - quote_oversize
        local_dataset_size = size // num_local_set
        local_size = {0: local_dataset_size+quote_oversize_single_node,
                      1: local_dataset_size,
                      2: local_dataset_size,
                      3: local_dataset_size,
                      4: local_dataset_size+quote_oversize_single_node
                      }#,
                      #5: local_dataset_size,
                      #6: local_dataset_size,
                      #7: local_dataset_size,
                      #8: local_dataset_size,
                      #9:local_dataset_size+quote_oversize_single_node
                      #}
    else:
        local_dataset_size = train_x.shape[0] // num_local_set
        local_size = {0: local_dataset_size,
                      1: local_dataset_size,
                      2: local_dataset_size,
                      3: local_dataset_size,
                      4: local_dataset_size}#,
                      #5: local_dataset_size,
                      #6: local_dataset_size,
                      #7: local_dataset_size,
                      #8: local_dataset_size,
                      #9:local_dataset_size}

    local_train_set = {0:[],
                       1:[],
                       2:[],
                       3:[],
                       4:[]}#,
                       #5:[],
                       #6:[],
                       #7:[],
                       #8:[],
                       #9:[]}

    local_train_label_set = {0:[],
                       1:[],
                       2:[],
                       3:[],
                       4:[]}#,
                       #5:[],
                       #6:[],
                       #7:[],
                       #8:[],
                       #9:[]}
    j = 0
    client = 0
    for i in range(len(local_train_set)):
        local_dataset_size = local_size[client]
        local_train_set[i] = train_x[j:j + local_dataset_size]
        local_train_label_set[i] = train_y[j:j + local_dataset_size]
        j = j + local_dataset_size
        client = client + 1

    #Shuffle i singoli set
    for i in range(len(local_train_set)):
        index_list = np.random.randint(local_train_set[i].shape[0], size=local_train_set[i].shape[0])
        random.shuffle(index_list)
        local_train_set[i] = local_train_set[i][index_list]
        local_train_label_set[i] = local_train_label_set[i][index_list]

    #print('Creazione Dataset Test: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
    test_x = ConvertString2List(test_data, features=3)
    #print('Dataset Creato Test: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
main()