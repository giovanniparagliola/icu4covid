import datetime
import random
import os
import tensorflow as tf
import pandas as pd
import re
import  numpy as np
from sklearn.model_selection import train_test_split

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

def getDataSet(nclient):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    train_path = dir_path + '/files/' + "ecg_dataset_trainingv6.csv"
    test_path = dir_path + '/files/' + "ecg_dataset_testv2.csv"
    dataraw = pd.read_csv(train_path, header=1, names=['samples', 'classes'])
    traingset_data = dataraw.samples.values
    training_class = dataraw.classes.values

    dataraw = pd.read_csv(test_path, header=1, names=['samples', 'classes'])
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


    test_x = ConvertString2List(test_data, features=3)

    print("Training Set Size: {}".format(train_x.shape))
    print("Validation Set Size:{}".format(val_x.shape))
    print("Test Set Size: {}".format(test_x.shape))

    #Esempio di segnale
    mask = train_y == 1
    mask = list(mask)
    indexs_high = mask.index(True)
    indexs_low = mask.index(False)

    sample = train_x[indexs_low]
    sample_df_low= pd.DataFrame(sample, columns=["V1","V2","V3"])
    sample_df_low.to_csv("sample_low.csv")

    sample_high = train_x[indexs_high]
    sample_df_high = pd.DataFrame(sample_high, columns=["V1", "V2", "V3"])
    sample_df_high.to_csv("sample_high.csv")
    num_local_set = 3

    local_dataset_size = train_x.shape[0] // num_local_set
    local_size = {0: local_dataset_size,1: local_dataset_size, 2: local_dataset_size}

    local_train_set = {0:[],1:[],2:[]}
    local_train_label_set = {0:[], 1:[], 2:[]}

    j = 0
    client = 0
    for i in range(len(local_train_set)):
        local_dataset_size = local_size[client]
        local_train_set[i] = train_x[j:j + local_dataset_size]
        local_train_label_set[i] = train_y[j:j + local_dataset_size]
        j = j + local_dataset_size
        client = client + 1
        print("Local Test Set Size: {}".format(local_train_set[i].shape))

    #Shuffle i singoli set
    for i in range(len(local_train_set)):
        index_list = np.random.randint(local_train_set[i].shape[0], size=local_train_set[i].shape[0])
        random.shuffle(index_list)
        local_train_set[i] = local_train_set[i][index_list]
        local_train_label_set[i] = local_train_label_set[i][index_list]
        return local_train_set[nclient],local_train_label_set[nclient], val_x, val_y , test_x, test_y
