import numpy as np
import pandas as pd
import random
#import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import BinaryAccuracy, Accuracy,Precision, Recall, AUC, SensitivityAtSpecificity, FalsePositives, FalseNegatives, TrueNegatives, TruePositives
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score,  roc_auc_score

def create_clients(image_list, label_list, num_clients=10, initial='clients'):
    ''' return: a dictionary with keys clients' names and value as
                data shards - tuple of images and label lists.
        args:
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1

    '''

    # create a list of client names
    client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]

    # randomize the data
    data = list(zip(image_list, label_list))
    random.shuffle(data)

    # shard data and place at each client
    size = len(data) // num_clients
    shards = [data[i:i + size] for i in range(0, size * num_clients, size)]

    # number of clients must equal number of shards
    assert (len(shards) == len(client_names))

    return {client_names[i]: shards[i] for i in range(len(client_names))}

def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    # seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)

def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    # get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    # first calculate the total training data points across clinets
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names]) * bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() * bs
    return local_count / global_count


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final

def mean_scaled_weights(scaled_weight_list):
    avg_grad = list()
    # get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_mean(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
    return avg_grad


def mean_weigths(scaled_weigth, weights=None):
    new_weights = list()
    for weights_list_tuple in zip(*scaled_weigth):
        if weights is None:
            new_weights.append(np.average(np.array(weights_list_tuple), axis=0))
        else:
            new_weights.append(np.average(np.array(weights_list_tuple), axis=0, weights=weights))
    return new_weights

def mediam_weigths(scaled_weigth):
    new_weights = list()
    for weights_list_tuple in zip(*scaled_weigth):
        new_weights.append(np.median(np.array(weights_list_tuple), axis=0))
    return new_weights

def mean_scaled_weigths(scaled_weigth):
    new_weights = list()
    for weights_list_tuple in zip(*scaled_weigth):
        new_weights.append(
            [np.array(weights_).mean(axis=0) \
             for weights_ in zip(*weights_list_tuple)])
    return new_weights

def evaluateModel(global_model,test_x, test_y,Hyperparameter,dir_path, id_job, client, idd_flag, args, num_of_parameters):
    print('\n# Evaluate on test data')
    results = global_model.evaluate(test_x, test_y, batch_size=Hyperparameter['batch_size'], verbose=0)
    print('test loss, test acc:', results)

    y_pred = global_model.predict(test_x)
    y_pred_temp = []
    y_pred_array = []

    for p in y_pred:
        y_pred_temp.append(p)
        if p > 0.5:
            y_pred_array.append(1)
        else:
            y_pred_array.append(0)
    y_pred_array = np.asarray(y_pred_array)
    y_pred_score = np.asarray(y_pred_temp)

    r = Recall()
    r.update_state(test_y, y_pred_array)
    print('Recall/Sensitivity: ', r.result().numpy())

    auc = roc_auc_score(test_y, y_pred_array)
    print('AUC:{}'.format(auc))

    fn = FalseNegatives()
    fn.update_state(test_y, y_pred_array)
    fn = fn.result().numpy()

    fp = FalsePositives()
    fp.update_state(test_y, y_pred_array)
    fp = fp.result().numpy()

    tn = TrueNegatives()
    tn.update_state(test_y, y_pred_array)
    tn = tn.result().numpy()

    tp = TruePositives()
    tp.update_state(test_y, y_pred_array)
    tp = tp.result().numpy()

    spec = tn / (tn + fp)
    print('Specificity: ', spec)

    ss = SensitivityAtSpecificity(spec, num_thresholds=1)
    ss.update_state(test_y, y_pred_array)
    #print('SensitivityAtSpecificity: ', ss.result().numpy())  # Final result: 0.5

    m = confusion_matrix(test_y, y_pred_array, labels=[1, 0])
    print("Confution Matrix:")
    print(m)

    print("TruePositive:{} \t TrueNegative:{} \t FalsePositive:{} \t FalseNegative:{}".format(tp, tn, fp, fn))

    fpr, tpr, thresholds = roc_curve(test_y, y_pred_score, pos_label=1)

    roc_value2 = {'fpr': fpr, 'tpr': tpr, 'thresds': thresholds}
    temp = pd.DataFrame(roc_value2)
    temp.to_csv(dir_path + '/roc/local_client_'+str(client)+'_roc_LCD' + str(id_job) + '.csv')

    f1 = f1_score(test_y, y_pred_array, average='binary')
    print("F1:{}".format(f1))
    #global_model.save("clientsmodels/local_"+client+"_model.h5")
    #print("Model size: {} Byte ".format(os.stat("local_"+client+"_model.h5").st_size))

    out = {'idJob':id_job,
           'IDD': args.decaymode,
           'DecayMode':args.decaymode,
           'FilterNode':args.filtermode,
           'RandonChoise':args.randomchoise,
           'TranceMode': args.tranchemode,
           'config':args.config,
           'num_of_parameters': num_of_parameters,
           'Client':client,
           'LOSS':results[0],
           'Acc':results[1],
           'Recall':r.result().numpy(),
           'AUC':auc,
           'Specificity':spec,
            'TP':tp,
           'TN':tn,
           'FP':fp,
           'FN':fn,
           'F1':f1}

    outfile = pd.DataFrame(out,index=[0])
    outfile.to_csv(dir_path + '/clients/'+str(client)+'/output_client_' + str(client) + '_job_' + str(id_job) + '.csv')
