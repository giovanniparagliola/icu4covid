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

def main(train_path, test_path):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--nClient', default=5, type=int)
    parser.add_argument('--rounds', default=20, type=int)
    parser.add_argument('--decaymode', default=0, type=int) #0= false , 1= true
    parser.add_argument('--tranchemode', default=0, type=int)
    parser.add_argument('--filtermode', default=0, type=int)
    parser.add_argument('--thr', default=0.80, type=float)
    parser.add_argument('--randomchoise', default=0, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--config', default=0, type=int)
    parser.add_argument('--idd', default=1, type=int) #0 no idd, 1 idd
    args = parser.parse_args()
    print(args)

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
    num_local_set = args.nClient

    idd_flag = False
    if args.idd == 1:
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

    #CONFIGURAZIONE DEL RUN
    comms_round = args.rounds
    thr = args.thr
    lr_decay_step = 5
    lr_decay = 5
    ratio = local_dataset_size // comms_round

    candidate_clientes_flag = False
    tranche_mode = False
    lr_decay_mode = False
    filter_candidate_clientes_flag = False

    if args.decaymode == 1:
        lr_decay_mode = True #args.decaymode

    if args.tranchemode==1: #args.tranchemode #quando è falso l'addestramento avviane su tutto il datase
        tranche_mode = True

    if args.filtermode == 1:
        candidate_clientes_flag = True

    if args.randomchoise==1:
        filter_candidate_clientes_flag = True

    num_of_parameters = 0
    list_dense_layers = []
    if args.config == 0:
        list_dense_layers = ["ldenso2"] #default
        num_of_parameters = 1049088
    elif args.config == 1:
        list_dense_layers = ["ldenso3", "ldenso4"]
        num_of_parameters = 131328 + 32896
    elif args.config == 2:
        list_dense_layers = ["ldenso1","ldenso3", "ldenso4"]
        num_of_parameters = 26624 + 131328 + 32896
    elif args.config == 3:
        list_dense_layers = ["ldenso1","ldenso2","ldenso3", "ldenso4"]
        num_of_parameters = 26624 + 1049088 + 131328 + 32896

    first_round = True
    only_danse_layers = True
    use_global_model = False

    id_job = random.Random().randrange(1, 1000, 1)
    print("ID Job:{}".format(id_job))
    Hyperparameter = {
        'lr': args.lr,
        'n_layer': 1,
        'lstm_outputs': 20,
        'n_hidden_fc': 2048,
        'batch_size': 32,
        'n_epoch': 2000,#2500
        'n_iteration': 1000, #1000
        'l2_penalty': 0.10,
        'kernel_size': 10, #10
        'filter_size': 8, #4
        'kernel_size_l2': 5,
        'filter_size_l2': 4, #2
        'pool_size': 10,
        'pool_size_l2': 10
    }

    timepoints = train_x.shape[1]
    #print("MY MODEL")
    global_model= am.LSTM_CNN_DNN_Flattten_v2(hp = Hyperparameter,
                                              timestap=timepoints,
                                              nfeatures=3) #am.CNN_LSTM_DNN_Flattten(hp = Hyperparameter, timestap=timepoints, nfeatures=3)
    global_model.compile(loss=keras.losses.BinaryCrossentropy(),
                         optimizer=keras.optimizers.Adam(),
                         metrics=[keras.metrics.binary_accuracy])
    global_model.summary()

    start=datetime.datetime.now()
    print('Inizio Addrestamento Rete: {:%Y-%m-%d %H:%M:%S}'.format(start))

    # commence global training loop
    last_local_weight_list = list() #inizializzo la lista con i pesi in modo casuale
    for i in range(len(local_train_set)):
        last_local_weight_list.append(global_model.get_weights())

    local_client_train_set = {0: [local_train_set[0][0]],
                              1: [local_train_set[0][0]],
                              2: [local_train_set[0][0]],
                              3: [local_train_set[0][0]],
                              4: [local_train_set[0][0]]}#,
                              #5: [local_train_set[0][0]],
                              #6: [local_train_set[0][0]],
                              #7: [local_train_set[0][0]],
                              #8: [local_train_set[0][0]],
                              #9: [local_train_set[0][0]]
                              #}

    local_client_train_label_set = {0: [local_train_label_set[0][0]],
                                    1: [local_train_label_set[0][0]],
                                    2: [local_train_label_set[0][0]],
                                    3: [local_train_label_set[0][0]],
                                    4: [local_train_label_set[0][0]]}#,
                                    #5: [local_train_label_set[0][0]],
                                    #6: [local_train_label_set[0][0]],
                                    #7: [local_train_label_set[0][0]],
                                    #8: [local_train_label_set[0][0]],
                                    #9: [local_train_label_set[0][0]]
                                    #}

    local_model_list = list()
    local_model_list_not_scaled = list()
    for comm_round in range(comms_round):
        print()
        if lr_decay_mode:
            if comm_round%lr_decay_step == 0:
                Hyperparameter['lr'] = Hyperparameter['lr']/lr_decay

        # get the global model's weights - will serve as the initial weights for all local models
        if (first_round == True):
            global_weights = global_model.get_weights()

        # initial list to collect local model weights after scalling
        scaled_local_weight_list = list()
        local_accuracy_list = list()

        only_canditated = True
        if first_round == False:
            previus_select = index_candidated_clients

        index_candidated_clients = list()

        #Liste per la gestione della medie dei layer
        dense_local_weight_list = list()
        index_dense_local_weight_list = list()

        # loop through each client and create new local model
        for client in range(0, num_local_set):
            #lista che gestisce i livelli densi del client corrente (necessario che sia in una lista per il calcolo della media)
            client_dense_local_weight_list = list()
            local_model = am.LSTM_CNN_DNN_Flattten_v2(hp = Hyperparameter,
                                                      timestap=timepoints,
                                                      nfeatures=3)
            local_model.compile(loss=keras.losses.BinaryCrossentropy(),
                                optimizer=keras.optimizers.Adam(),
                                metrics=[keras.metrics.binary_accuracy])

            callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
            # set local model weight to the weight of the global model

            if only_canditated == False:
                if use_global_model:
                    local_model.set_weights(global_weights)  # parte originale
                elif first_round == False:
                    if only_danse_layers == False:
                        local_model.set_weights(local_model_list[client].get_weights())  # Carico tutto il modello
                    else:  # aggiorno solo i livelli densi
                        for i in index_dense_local_weight_list:
                            local_model.get_weights()[i] = dict_layers[i]
                else:
                    #in questyo modo tutti i client hanno la stessa inizializzazione
                    local_model.set_weights(global_weights)  # parte originale
            else:
                if use_global_model:
                    if first_round == False:
                        if client in previus_select:
                            local_model.set_weights(global_weights)  # parte originale
                        else:
                            local_model.set_weights(local_model_list_not_scaled[client])
                    else:
                        local_model.set_weights(global_weights)  # parte originale
                elif first_round == False:
                    if client in previus_select:
                        if only_danse_layers == False:
                            # print(len(local_model_list))
                            local_model.set_weights(local_model_list[client].get_weights())  # Carico tutto il modello
                        else:  # aggiorno solo i livelli densi
                            for i in index_dense_local_weight_list:
                                local_model.get_weights()[i] = dict_layers[i]
                    else:
                        local_model.set_weights(local_model_list_not_scaled[client].get_weights())  # Carico tutto il modello
                else:
                    # in questyo modo tutti i client hanno la stessa inizializzazione
                    local_model.set_weights(global_weights)  # parte originale

            # fit local model with client's data
            #ad ogni randund glido una porzione di dataset diverso
            if tranche_mode:
                ratio = local_size[client] // comms_round
                local_train_set[client], X_local_train_, local_train_label_set[client], y_local_train_ = random_set(local_train_set[client], local_train_label_set[client], ratio)
                local_client_train_set[client] = np.concatenate((local_client_train_set[client], X_local_train_))
                local_client_train_label_set[client] = np.concatenate((local_client_train_label_set[client], y_local_train_))
                X_local_train = local_client_train_set[client]
                y_local_train = local_client_train_label_set[client]
            else:
                X_local_train = local_train_set[client]
                y_local_train = local_train_label_set[client]

            history = local_model.fit(X_local_train, y_local_train,
                                      batch_size = Hyperparameter['batch_size'],
                                      epochs=Hyperparameter['n_epoch'],
                                      validation_data=(val_x, val_y),
                                      verbose=0,
                                      callbacks=[callback]) #, batch_size=None Hyperparameter['batch_size']
            #pezzoot
            if ((client==0) and comm_round==0):
                local_model.save_weights("local1model")
            if (comm_round % 10 == 0) or (comm_round == comms_round - 1):  # saldo ogni 5 round e l'ultimo round
                local_history = pd.DataFrame(history.history)
                local_history.to_csv(dir_path+'/clients/'+str(client)+'/'+ str(comm_round)+"_history_"+str(id_job)+'.csv')
            results = local_model.evaluate(test_x, test_y, batch_size=Hyperparameter['batch_size'], verbose=0)
            print('Round:'+str(comm_round) + ' Client: ' + str(client) + ' test loss, test acc:', results)
            print("Spape:({} , {})".format(X_local_train.shape[0], X_local_train.shape[1]))
            local_accuracy_list.append(results[1])
            # scale the model weights and add to list

        #Logica per la media solo sul livello denso
            if use_global_model == False:
                i = 0
                for l in local_model.weights:
                    if l.name in list_dense_layers == True: #l.name.find("ldenso2") != -1:
                        # temp = local_model.get_weights()[i] * 0.3
                        factor = X_local_train.shape[0]/train_x.shape[0]
                        scaled_local_weigth = scale_model_weights(local_model.get_weights()[i],factor )
                        client_dense_local_weight_list.append(scaled_local_weigth)  # salvo solo i pnsi dei livelli densi
                        index_dense_local_weight_list.append(i)
                    i = i + 1
                dense_local_weight_list.append(client_dense_local_weight_list)
                if first_round:
                    local_model_list.append(local_model)  # qui metto il modello locale addrestrato
                    local_model_list_not_scaled.append(local_model)
                else:
                    local_model_list[client] = local_model  # qui sostituisco il modello locale addrestrato
                    local_model_list_not_scaled[client] = local_model
            else:
                #Modello globale - prendo tutti i pesi del modello locoale
                factor = X_local_train.shape[0] / train_x.shape[0]
                scaled_local_weigth = scale_model_weights(local_model.get_weights(), factor)
                scaled_local_weight_list.append(scaled_local_weigth) #iid
                #scaled_local_weight_list.append(scaled_weights)  # no-iid

            #last_local_weight_list[client] = local_model.get_weights()

            # clear session to free memory after each communication round
            K.clear_session()

        # to get the average over all the local model, we simply take the sum of the scaled weights
        i = 0
        #Usoun sottoinsieme dei nodi che hanon avuto un addrestramento positivo
        if candidate_clientes_flag:
            for elem in local_accuracy_list:
                if elem > thr:
                    index_candidated_clients.append(i)
                i = i+1

            if len(index_candidated_clients) != 0:
                if filter_candidate_clientes_flag:
                    num_index = (len(index_candidated_clients) // 2) + 1
                    index_candidated_clients = np.random.choice(index_candidated_clients, size=num_index, replace=False)
                if use_global_model:
                    random_scaled_local_weight_list = [scaled_local_weight_list[i] for i in index_candidated_clients]
                    scaled_local_weight_list= random_scaled_local_weight_list
                else:
                    random_scaled_local_weight_list = [dense_local_weight_list[i] for i in index_candidated_clients]
                    dense_local_weight_list = random_scaled_local_weight_list

        if use_global_model:
            average_weights = mean_scaled_weights(scaled_local_weight_list)
            std_weights = std_scaled_weights(scaled_local_weight_list)
            # update global model
            global_model.set_weights(average_weights)
            # test global model and print out metrics after each communications round
            results = global_model.evaluate(test_x, test_y, batch_size=Hyperparameter['batch_size'], verbose=0)
            print('Global test loss, test acc:', results)
        else:
            first_round = False
            std_weights = std_scaled_weights(dense_local_weight_list)
            dense_average_weights = mean_scaled_weights(dense_local_weight_list)
            dict_layers = dict(zip(index_dense_local_weight_list, dense_average_weights))
            j = 0
            for m in local_model_list:
                for i in index_dense_local_weight_list:
                    m.get_weights()[i] = dict_layers[i]
                local_model_list[j] = m
                j = j+1

    global_model.save_weights("global_model", save_format="tf")
    print('Fine  Addrestamento Rete: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

    #training_history = pd.DataFrame(history.history)
    #training_history.to_csv('results_LCD.csv')
    #print('\nhistory dict:', history.history)
    if use_global_model:
        print('\n# Evaluate on test data')
        results = global_model.evaluate(test_x, test_y, batch_size=Hyperparameter['batch_size'], verbose=0)
        print('test loss, test acc:', results)

        y_pred = global_model.predict(test_x)
        y_pred_temp = []
        y_pred_array = []

        for p in y_pred:
            y_pred_temp.append(p)
            if p> 0.5:
                y_pred_array.append(1)
            else:
                y_pred_array.append(0)
        y_pred_array= np.asarray(y_pred_array)
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
        print('SensitivityAtSpecificity: ', ss.result().numpy())  # Final result: 0.5

        m = confusion_matrix(test_y, y_pred_array, labels=[1,0])
        print("Confution Matrix:" )
        print(m)

        print("TruePositive:{} \t TrueNegative:{} \t FalsePositive:{} \t FalseNegative:{}".format(tp, tn, fp, fn))

        fpr, tpr, thresholds = roc_curve(test_y, y_pred_score, pos_label=1)

        roc_value2 = {'fpr': fpr, 'tpr': tpr, 'thresds': thresholds}
        temp = pd.DataFrame(roc_value2)
        temp.to_csv(dir_path+'/global_roc_LCD'+ str(id_job) +'.csv')

        f1=f1_score(test_y, y_pred_array, average='binary')
        print("F1:{}".format(f1))
        #global_model.save("global_model.h5")
        #print("Model size: {} Byte ".format(os.stat('global_model.h5').st_size))
    else:
        i=0
        for m in local_model_list:
            print("CLIENT:{}".format(i))
            evaluateModel(m, test_x, test_y, Hyperparameter, dir_path, id_job, i, idd_flag, args, num_of_parameters)
            print()
            print()
            i = i + 1

    #evaluation_results={ 'test_loss': results[0], 'test_acc': float(results[1]), 'precision':float(precision.result().numpy()), 'recall': float(recall.result().numpy()), 'accuracy':float(accuracy.result().numpy())}
    #evaluation_history = pd.DataFrame(evaluation_results, index=[0])
    #evaluation_history.to_csv('evaluation_results.csv')
    #auc = keras.metrics.AUC(num_thresholds=2)
    #auc.update_state(test_y_list, y_pred)
    #print('AUC result: ', auc.result().numpy())  # Final result: 0.75


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_train = dir_path + '/files/' + "ecg_dataset_trainingv6.csv"
    path_test = dir_path + '/files/' + "ecg_dataset_testv2.csv"
    main(train_path=path_train, test_path=path_test)


'''
            if local_average_flag:
                if comm_round == 0: #è la prima iterazione
                    local_model.set_weights(global_weights) # parte originale

                else: #le successive medio il globale con quello attuale per "avvicinare" al cio che avevo prima...
                    temporal_local_accuracy_list = list()
                    temporal_local_accuracy_list.append(global_weights)
                    temporal_local_accuracy_list.append(last_local_weight_list[client])
                    local_average_weights = mean_scaled_weights(temporal_local_accuracy_list)
                    local_model.set_weights(local_average_weights)
            else:
'''