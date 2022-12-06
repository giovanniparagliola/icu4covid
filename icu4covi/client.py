import flwr as fl
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
dir_path = os.path.dirname(os.path.realpath(__file__))
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
parser = argparse.ArgumentParser(description='')
parser.add_argument('--nClient', default=0, type=int)
args = parser.parse_args()
print("Data Loading...")
x_train, y_train, x_val, y_val, x_test, y_test = getDataSet(args.nClient)
print("Data Loaded Complete")
Hyperparameter = {
    'lr': 0.0001,
    'n_layer': 1,
    'lstm_outputs': 20,
    'n_hidden_fc': 2048,
    'batch_size': 32,
    'n_epoch': 1,  # 2500
    'n_iteration': 1,  # 1000
    'l2_penalty': 0.10,
    'kernel_size': 10,  # 10
    'filter_size': 8,  # 4
    'kernel_size_l2': 5,
    'filter_size_l2': 4,  # 2
    'pool_size': 10,
    'pool_size_l2': 10
}
timepoints=300
model = am.LSTM_CNN_DNN_Flattten_v2(hp=Hyperparameter,
                                           timestap=timepoints,
                                           nfeatures=3)  # am.CNN_LSTM_DNN_Flattten(hp = Hyperparameter, timestap=timepoints, nfeatures=3)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                     optimizer=tf.keras.optimizers.Adam(),
                     metrics=[tf.keras.metrics.binary_accuracy])

#model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
#model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])


class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": float(accuracy)}

fl.client.start_numpy_client(server_address="172.16.1.21:8080", client=CifarClient())