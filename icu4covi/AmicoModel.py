from tensorflow import keras


def CNN_LSTM_DNN(outs =1, hp=None, binary_classification=True, timestap=310, nfeatures=4):
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(filters = hp['filter_size'],
                     kernel_size= hp['kernel_size'],
                     input_shape=(timestap, nfeatures),
                     batch_size=hp['batch_size'],
                     padding='causal'
                     ))
    model.add(keras.layers.MaxPooling1D(pool_size=hp['pool_size']))
    model.add(keras.layers.LSTM(hp['lstm_outputs'], return_sequences=True))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=hp['n_hidden_fc'],
                    kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                    kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='normal', seed=None),
                    activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    n_hidden_fc2 = (hp['n_hidden_fc']) // 2
    model.add(keras.layers.Dense(units=n_hidden_fc2,
                    kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                    kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='normal', seed=None)
                    ))
    model.add(keras.layers.Dense(units=outs,
              kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
              kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='normal',
                                                              seed=None)
            ))
    if binary_classification:
        model.add(keras.layers.Activation('sigmoid'))
    else:
        model.add(keras.layers.Activation('softmax'))
    return model


def CNN_LSTM_DNN_Flattten(outs =1, hp=None, bin=True, timestap=310, nfeatures=4):
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(hp['filter_size'],
                     hp['kernel_size'],
                     input_shape=(timestap, nfeatures),
                     batch_size=hp['batch_size'],
                     padding='causal'
                     ))
    model.add(keras.layers.MaxPooling1D(pool_size=hp['pool_size']))
    model.add(keras.layers.Conv1D(hp['filter_size_l2'],
                     hp['kernel_size_l2'],
                     batch_size=hp['batch_size'],
                     padding='causal'
                     ))
    model.add(keras.layers.MaxPooling1D(pool_size=hp['pool_size_l2']))
    model.add(keras.layers.LSTM(hp['lstm_outputs'], activation = "sigmoid", return_sequences=True))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=hp['n_hidden_fc'],
                    kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                    kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='normal', seed=None),
                    activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())
    n_hidden_fc2 = (hp['n_hidden_fc']) // 2
    model.add(keras.layers.Dense(units=n_hidden_fc2,
                    kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                    kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='normal', seed=None)
                    ))
    model.add(keras.layers.BatchNormalization())
    n_hidden_fc3 = (hp['n_hidden_fc']) // 4
    model.add(keras.layers.Dense(units=n_hidden_fc3,
                                 kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                       distribution='normal', seed=None)
                                 ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units=outs,
              kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
              kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='normal',
                                                              seed=None)
            ))
    if bin:
        model.add(keras.layers.Activation('sigmoid'))
    else:
        model.add(keras.layers.Activation('softmax'))
    return model

def LSTM_CNN_DNN_Flattten(hp=None,  timestap=310, nfeatures=4):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(hp['lstm_outputs'],
                                input_shape= (timestap, nfeatures),
                                activation="tanh",
                                recurrent_activation = 'sigmoid',
                                return_sequences=True))
    #model.add(keras.layers.Flatten())
    model.add(keras.layers.Conv1D(hp['filter_size'],
                                  hp['kernel_size'],
                                  batch_size=hp['batch_size'],
                                  padding='causal'
                                  ))
    model.add(keras.layers.MaxPooling1D(pool_size=hp['pool_size']))
    model.add(keras.layers.Conv1D(hp['filter_size_l2'],
                                  hp['kernel_size_l2'],
                                  batch_size=hp['batch_size'],
                                  padding='causal'
                                  ))
    model.add(keras.layers.MaxPooling1D(pool_size=hp['pool_size_l2']))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=hp['n_hidden_fc'],
                                 kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                       distribution='normal',
                                                                                       seed=None),
                                 activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())
    n_hidden_fc2 = (hp['n_hidden_fc']) // 4
    model.add(keras.layers.Dense(units=n_hidden_fc2,
                                 kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                       distribution='normal', seed=None)
                                 ))
    model.add(keras.layers.BatchNormalization())
    n_hidden_fc3 = (hp['n_hidden_fc']) // 8
    model.add(keras.layers.Dense(units=n_hidden_fc3,
                                 kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                       distribution='normal', seed=None)
                                 ))
    n_hidden_fc4 = (hp['n_hidden_fc']) // 16
    model.add(keras.layers.Dense(units=n_hidden_fc4,
                                 kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                       distribution='normal', seed=None)
                                 ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units=1,
                                 kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                       distribution='normal',
                                                                                       seed=None)
                                 ))
    model.add(keras.layers.Activation('sigmoid'))
    return model


def LSTM_CNN_DNN_Flattten_v2(hp=None,  timestap=310, nfeatures=4):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(hp['lstm_outputs'],
                                input_shape= (timestap, nfeatures),
                                activation="tanh",
                                recurrent_activation = 'sigmoid',
                                return_sequences=True
                                ))
    #model.add(keras.layers.Flatten())
    model.add(keras.layers.Conv1D(hp['filter_size'],
                                  hp['kernel_size'],
                                  batch_size=hp['batch_size'],
                                  padding='causal'
                                  ))
    model.add(keras.layers.MaxPooling1D(pool_size=hp['pool_size']))
    model.add(keras.layers.Conv1D(hp['filter_size_l2'],
                                  hp['kernel_size_l2'],
                                  batch_size=hp['batch_size'],
                                  padding='causal'
                                  ))
    model.add(keras.layers.MaxPooling1D(pool_size=hp['pool_size_l2']))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=hp['n_hidden_fc'], name="ldenso1",
                                 kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                       distribution='normal',
                                                                                       seed=None),
                                 activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())
    n_hidden_fc2 = (hp['n_hidden_fc']) // 4
    model.add(keras.layers.Dense(units=n_hidden_fc2, name="ldenso2",
                                 kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                       distribution='normal', seed=None)
                                 ))
    model.add(keras.layers.BatchNormalization())
    n_hidden_fc3 = (hp['n_hidden_fc']) // 8
    model.add(keras.layers.Dense(units=n_hidden_fc3, name="ldenso3",
                                 kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                       distribution='normal', seed=None)
                                 ))
    n_hidden_fc4 = (hp['n_hidden_fc']) // 16
    model.add(keras.layers.Dense(units=n_hidden_fc4, name="ldenso4",
                                 kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                       distribution='normal', seed=None)
                                 ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units=1, name="ldenso5",
                                 kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                       distribution='normal',
                                                                                       seed=None)
                                 ))
    model.add(keras.layers.Activation('sigmoid'))
    return model


def LSTM_CNN_DNN_Flattten_v3(hp=None,  timestap=310, nfeatures=4):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(hp['lstm_outputs'],
                                input_shape= (timestap, nfeatures),
                                activation="tanh",
                                recurrent_activation = 'sigmoid',
                                return_sequences=True
                                ))
    #model.add(keras.layers.Flatten())
    model.add(keras.layers.Conv1D(hp['filter_size'],
                                  hp['kernel_size'],
                                  batch_size=hp['batch_size'],
                                  padding='causal'
                                  ))
    model.add(keras.layers.MaxPooling1D(pool_size=hp['pool_size']))
    model.add(keras.layers.Conv1D(hp['filter_size_l2'],
                                  hp['kernel_size_l2'],
                                  batch_size=hp['batch_size'],
                                  padding='causal'
                                  ))
    model.add(keras.layers.MaxPooling1D(pool_size=hp['pool_size_l2']))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=hp['n_hidden_fc'], name="ldenso1",
                                 kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                       distribution='normal',
                                                                                       seed=None),
                                 activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dense(units=hp['n_hidden_fc']//2, name="ldenso2",
                                 kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                       distribution='normal',
                                                                                       seed=None),
                                 activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dense(units=hp['n_hidden_fc']//2, name="ldenso3",
                                 kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                       distribution='normal',
                                                                                       seed=None),
                                 activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())

    n_hidden_fc2 = (hp['n_hidden_fc']) // 4
    model.add(keras.layers.Dense(units=n_hidden_fc2, name="ldenso4",
                                 kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                       distribution='normal', seed=None)
                                 ))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())
    n_hidden_fc3 = (hp['n_hidden_fc']) // 8

    model.add(keras.layers.Dense(units=n_hidden_fc3, name="ldenso5",
                                 kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                       distribution='normal', seed=None)
                                 ))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())

    n_hidden_fc4 = (hp['n_hidden_fc']) // 16
    model.add(keras.layers.Dense(units=n_hidden_fc4, name="ldenso6",
                                 kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                       distribution='normal', seed=None)
                                 ))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dense(units=1, name="ldenso7",
                                 kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                       distribution='normal',
                                                                                       seed=None)
                                 ))
    model.add(keras.layers.Activation('sigmoid'))
    return model



def LSTM_AE_CNN_DNN_Flattten(hp=None,  timestap=310, nfeatures=4):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(hp['lstm_outputs'],
                                input_shape= (timestap, nfeatures),
                                activation="tanh",
                                recurrent_activation = 'sigmoid',
                                return_sequences=True))
    #model.add(keras.layers.Flatten())
    model.add(keras.layers.Conv1D(hp['filter_size'],
                                  hp['kernel_size'],
                                  batch_size=hp['batch_size'],
                                  padding='causal'
                                  ))
    model.add(keras.layers.MaxPooling1D(pool_size=hp['pool_size']))
    model.add(keras.layers.Conv1D(hp['filter_size_l2'],
                                  hp['kernel_size_l2'],
                                  batch_size=hp['batch_size'],
                                  padding='causal'
                                  ))
    model.add(keras.layers.MaxPooling1D(pool_size=hp['pool_size_l2']))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=hp['n_hidden_fc'],
                                 kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                       distribution='normal',
                                                                                       seed=None),
                                 activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())
    n_hidden_fc2 = (hp['n_hidden_fc']) // 4
    model.add(keras.layers.Dense(units=n_hidden_fc2,
                                 kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                       distribution='normal', seed=None)
                                 ))
    model.add(keras.layers.BatchNormalization())
    n_hidden_fc3 = (hp['n_hidden_fc']) // 8
    model.add(keras.layers.Dense(units=n_hidden_fc3,
                                 kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                       distribution='normal', seed=None)
                                 ))
    n_hidden_fc4 = (hp['n_hidden_fc']) // 16
    model.add(keras.layers.Dense(units=n_hidden_fc4,
                                 kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                       distribution='normal', seed=None)
                                 ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units=1,
                                 kernel_regularizer=keras.regularizers.l2(hp['l2_penalty']),
                                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg',
                                                                                       distribution='normal',
                                                                                       seed=None)
                                 ))
    model.add(keras.layers.Activation('sigmoid'))
    return model

 #Modelli di addestramento alternati

def FCN(timestap=300, nfeatures=3):
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(128, 8,  input_shape= (timestap, nfeatures),padding='same', kernel_initializer='he_uniform'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation='relu'))
    model.add(keras.layers.Conv1D(256, 5, padding='same', kernel_initializer='he_uniform'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation='relu'))
    model.add(keras.layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation='relu'))
    model.add((keras.layers.GlobalAveragePooling1D()))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    return model

def Resnet(timestap, nfeatures, n_feature_maps, nb_classes=2):
    print('build conv_x')
    x = keras.layers.Input(shape=(timestap,nfeatures))
    conv_x = keras.layers.BatchNormalization()(x)
    conv_x = keras.layers.Conv1D(n_feature_maps, 8, 1, padding='same')(conv_x)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    print('build conv_y')
    conv_y = keras.layers.Conv1D(n_feature_maps, 5, 1, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    print('build conv_z')
    conv_z = keras.layers.Conv1D(n_feature_maps, 3, 1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    #is_expand_channels = not (input_shape[-1] == n_feature_maps)
    #if is_expand_channels:
    shortcut_y = keras.layers.Conv1D(n_feature_maps, 1, 1, padding='same')(x)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    #else:
    #    shortcut_y = keras.layers.BatchNormalization()(x)
    #shortcut_y = keras.layers.BatchNormalization()(x)
    print('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)

    print('build conv_x')
    x1 = y
    conv_x = keras.layers.Conv1D(n_feature_maps * 2, 8, 1, padding='same')(x1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    print('build conv_y')
    conv_y = keras.layers.Conv1D(n_feature_maps * 2, 5, 1, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    print('build conv_z')
    conv_z = keras.layers.Conv1D(n_feature_maps * 2, 3, 1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    #is_expand_channels = not (input_shape[-1] == n_feature_maps * 2)
    #if is_expand_channels:
    shortcut_y = keras.layers.Conv1D(n_feature_maps * 2, 1, 1, padding='same')(x1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    #else:
    #    shortcut_y = keras.layers.BatchNormalization()(x1)

    #shortcut_y = keras.layers.BatchNormalization()(x1)
    print('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)

    print('build conv_x')
    x1 = y
    conv_x = keras.layers.Conv1D(n_feature_maps * 2, 8, 1, padding='same')(x1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    print('build conv_y')
    conv_y = keras.layers.Conv1D(n_feature_maps * 2, 5, 1, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    print('build conv_z')
    conv_z = keras.layers.Conv1D(n_feature_maps * 2, 3, 1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    #is_expand_channels = not (input_shape[-1] == n_feature_maps * 2)
    #if is_expand_channels:
    shortcut_y = keras.layers.Conv1D(n_feature_maps * 2, 1, 1, padding='same')(x1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    #else:
    #    shortcut_y = keras.layers.BatchNormalization()(x1)

    #shortcut_y = keras.layers.BatchNormalization()(x1)
    print('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)

    full = keras.layers.GlobalAveragePooling1D()(y)
    out = keras.layers.Dense(nb_classes, activation='softmax')(full)
    print('        -- model was built.')
    model = keras.models.Model(inputs=x, outputs=out)
    return model

def Encoder(timestap=300, nfeatures=3):
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(128, 5, input_shape=(timestap, nfeatures), padding='same', kernel_initializer='he_uniform'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation='relu'))
    model.add(keras.layers.Conv1D(256, 11, padding='same', kernel_initializer='he_uniform'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation='relu'))
    model.add(keras.layers.Conv1D(128, 21, padding='same', kernel_initializer='he_uniform'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation='relu'))
    model.add(keras.layers.PReLU())
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.GlobalMaxPooling1D())
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    return model

#RIVEDI causa molto lavor odi preprocessing
def MCNN(timestap=300, nfeatures=3):
    # main CNN model - CNN1
    main_model = keras.Sequential()
    main_model.add(keras.layers.Conv1D(32, 3, 3, input_shape=(timestap,nfeatures)))
    main_model.add(keras.layers.Activation('relu'))
    main_model.add(keras.layers.MaxPooling1D(pool_size=2))

    main_model.add(keras.layers.Conv1D(32, 3, 3))
    main_model.add(keras.layers.Activation('relu'))
    main_model.add(keras.layers.MaxPooling1D(pool_size=2))

    main_model.add(keras.layers.Conv1D(64, 3, 3))
    main_model.add(keras.layers.Activation('relu'))
    main_model.add(keras.layers.MaxPooling1D(pool_size=2))  # the main_model so far outputs 3D feature maps (height, width, features)

    main_model.add(keras.layers.Flatten())

    # lower features model - CNN2
    lower_model1 = keras.Sequential()
    lower_model1.add(keras.layers.Conv1D(32, 3, 3, input_shape=(timestap,nfeatures)))
    lower_model1.add(keras.layers.Activation('relu'))
    lower_model1.add(keras.layers.MaxPooling1D(pool_size=2))
    lower_model1.add(keras.layers.Flatten())

    # lower features model - CNN3
    lower_model2 = keras.Sequential()
    lower_model2.add(keras.layers.Conv1D(32, 3, 3, input_shape=(timestap, nfeatures)))
    lower_model2.add(keras.layers.Activation('relu'))
    lower_model2.add(keras.layers.MaxPooling1D(pool_size=2))
    lower_model2.add(keras.layers.Flatten())

    # merged model
    merged_model = keras.layers.Concatenate([main_model, lower_model1, lower_model2])

    final_model = keras.Sequential()
    final_model.add(merged_model)
    final_model.add(keras.layers.Dense(64))
    final_model.add(keras.layers.Activation('relu'))
    final_model.add(keras.layers.Dropout(0.5))
    final_model.add(keras.layers.Dense(1))
    final_model.add(keras.layers.Activation('sigmoid'))
    final_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return final_model

def MCDCNN(timestap=300):
    #first channel
    input1=keras.layers.Input(shape=(timestap,1))
    conv11 = keras.layers.Conv1D(filters=8, kernel_size = 128)(input1)
    maxpool11 = keras.layers.MaxPooling1D(2)(conv11)
    conv12 = keras.layers.Conv1D(filters=5, kernel_size = 8)(maxpool11)
    maxpool12 = keras.layers.MaxPooling1D(2)(conv12)
    flatten1 = keras.layers.Flatten()(maxpool12)

    #second channel
    input2 = keras.layers.Input(shape=(timestap,1))
    conv21 = keras.layers.Conv1D(filters=8, kernel_size=128)(input2)
    maxpool21 = keras.layers.MaxPooling1D(2)(conv21)
    conv22 = keras.layers.Conv1D(filters=5, kernel_size=8)(maxpool21)
    maxpool22 = keras.layers.MaxPooling1D(2)(conv22)
    flatten2 = keras.layers.Flatten()(maxpool22)

    #third channel
    # second channel
    input3 = keras.layers.Input(shape=(timestap,1))
    conv31 = keras.layers.Conv1D(filters=8, kernel_size=128)(input3)
    maxpool31 = keras.layers.MaxPooling1D(2)(conv31)
    conv32 = keras.layers.Conv1D(filters=5, kernel_size=8)(maxpool31)
    maxpool32 = keras.layers.MaxPooling1D(2)(conv32)
    flatten3 = keras.layers.Flatten()(maxpool32)

    merged = keras.layers.Concatenate()
    merged = merged([flatten1, flatten2, flatten3])
    fc = keras.layers.Dense(units=800) (merged)
    act = keras.layers.Activation(activation="relu")(fc)
    out = keras.layers.Dense(units=1, activation="sigmoid")(act)
    model = keras.Model(inputs=[input1, input2, input3], outputs=out)
    return model

def TCNN(timestap=300, nfeatures=3):
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(128, 6,  input_shape= (timestap, nfeatures),padding='same', kernel_initializer='he_uniform'))
    model.add(keras.layers.AveragePooling1D(3))
    model.add(keras.layers.Activation(activation='sigmoid'))
    model.add(keras.layers.Conv1D(256, 12, padding='same', kernel_initializer='he_uniform'))
    model.add(keras.layers.AveragePooling1D(3))
    model.add(keras.layers.Activation(activation='sigmoid'))
    model.add(keras.layers.Dense(units=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    return model

def LeNet(timestap=300, nfeatures=3):
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(128, 5, input_shape=(timestap, nfeatures), padding='same', kernel_initializer='he_uniform'))
    model.add(keras.layers.MaxPooling1D(2))
    model.add(keras.layers.Activation(activation='relu'))
    model.add(keras.layers.Conv1D(256, 20, padding='same', kernel_initializer='he_uniform'))
    model.add(keras.layers.MaxPooling1D(4))
    model.add(keras.layers.Activation(activation='relu'))
    model.add((keras.layers.Dense(units=500, activation="relu")))
    model.add(keras.layers.MaxPooling1D(2))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    return model

