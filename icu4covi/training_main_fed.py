import AmicoModel as am
import datetime
import random
import os
import tensorflow as tf
from data_utility import *
from tensorflow import keras
#from fl_utils import *
import argparse
import warnings
warnings.filterwarnings('ignore')


physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
try:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

dir_path = os.path.dirname(os.path.realpath(__file__))


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--nClient', default=0, type=int)
    args = parser.parse_args()

    train_x, train_y, val_x, val_y, test_x, test_y = getDataSet(args.nClient)
    print("Training Set Size: {}".format(train_x.shape))
    print("Validation Set Size:{}".format(val_x.shape))
    print("Test Set Size: {}".format(test_x.shape))
    #CONFIGURAZIONE DEL RUN
    comms_round = 10

    id_job = random.Random().randrange(1, 1000, 1)
    print("ID Job:{}".format(id_job))
    Hyperparameter = {
        'lr': 0.0001,
        'n_layer': 1,
        'lstm_outputs': 20,
        'n_hidden_fc': 2048,
        'batch_size': 32,
        'n_epoch': 1,#2500
        'n_iteration': 1, #1000
        'l2_penalty': 0.10,
        'kernel_size': 10, #10
        'filter_size': 8, #4
        'kernel_size_l2': 5,
        'filter_size_l2': 4, #2
        'pool_size': 10,
        'pool_size_l2': 10
    }

    timepoints = train_x.shape[1]
    print("timepoints{}".format(timepoints))
    #print("MY MODEL")
    global_model= am.LSTM_CNN_DNN_Flattten_v2(hp = Hyperparameter,
                                              timestap=timepoints,
                                              nfeatures=3) #am.CNN_LSTM_DNN_Flattten(hp = Hyperparameter, timestap=timepoints, nfeatures=3)
    global_model.compile(loss=keras.losses.BinaryCrossentropy(),
                         optimizer=keras.optimizers.Adam(),
                         metrics=[keras.metrics.binary_accuracy])
    #global_model.summary()

    start=datetime.datetime.now()
    print('Inizio Addrestamento Rete: {:%Y-%m-%d %H:%M:%S}'.format(start))


    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
    # set local model weight to the weight of the global model

    history = global_model.fit(train_x, train_y,
                                      batch_size = Hyperparameter['batch_size'],
                                      epochs=Hyperparameter['n_epoch'],
                                      validation_data=(val_x, val_y),
                                      verbose=0,
                                      callbacks=[callback]) #, batch_size=None Hyperparameter['batch_size']
    accuracy = history.history["binary_accuracy"][0]
    loss, accuracy = global_model.evaluate(test_x, test_y)
    print("accuracy{}:".format(float(accuracy)))

    '''
    local_model.compile(loss=keras.losses.BinaryCrossentropy(),
                                optimizer=keras.optimizers.Adam(),
                                metrics=[keras.metrics.binary_accuracy])

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
            # set local model weight to the weight of the global model

    history = global_model.fit(train_x, train_y,
                                      batch_size = Hyperparameter['batch_size'],
                                      epochs=Hyperparameter['n_epoch'],
                                      validation_data=(val_x, val_y),
                                      verbose=0,
                                      callbacks=[callback]) #, batch_size=None Hyperparameter['batch_size']
    '''

if __name__ == "__main__":
    main()
