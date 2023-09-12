

def CNNs_train_modify(x_train, y_train, hyper_param):
    '''
    convolution neural network, 
    refer to Ho et al. 2019
    
    Parameters:
    ==========
    x_train:
    y_train:
    hyper_param: this is general parameters used for modelling 
        hyper_param = (iterats, learning_rate, decay)
        
        iterats: epoch for iteration
        learning_rate: learning rate
        decay: decay rate for learning rate
    
    ===============
    '''

    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers,regularizers
    from tensorflow.keras import initializers
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    import random
    import os

    
    iterats, learning_rate, decay = hyper_param
    #set random seed
    tf.random.set_seed(11)
    random.seed(11)
    os.environ['PYTHONHASHSEED'] = '12'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1' 
    np.random.seed(11)
    tf.keras.utils.set_random_seed(11)
    tf.config.experimental.enable_op_determinism()

    def my_model(mtrain):
    
        inputs = keras.Input(shape=mtrain)
        x = layers.ZeroPadding2D(padding=(2, 2))(inputs)
        x = layers.Conv2D(filters=24, kernel_size=5, padding='valid',
                   activation='relu', kernel_regularizer=l2(decay),
                         kernel_initializer=initializers.glorot_uniform(seed=35))(x)
        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(filters=10, kernel_size=3, padding='valid',
                   activation='relu', kernel_regularizer=l2(decay),
                         kernel_initializer=initializers.glorot_uniform(seed=123))(x)
        
        x = layers.MaxPooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=128, activation='relu',
                  kernel_regularizer=l2(decay),kernel_initializer=initializers.glorot_uniform(seed=125))(x)
        x = layers.Dense(units=64, activation='relu',
                  kernel_regularizer=l2(decay),kernel_initializer=initializers.glorot_uniform(seed=127))(x)
        x = layers.Dense(units=1, activation='relu',
                  kernel_regularizer=l2(decay),kernel_initializer=initializers.glorot_uniform(seed=121))(x)

        model = Model(inputs, x)
        return model

    mtrain = x_train.shape[1:]
    model = my_model(mtrain)
    print(model.summary())
    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    # opt = keras.optimizers.SGD(lr=learning_rate, momentum=0.9)
    
    model.compile(loss='mean_squared_error', optimizer=opt,
                  metrics=['mean_squared_error'])
    
    def lr_time_based_decay(epoch, lr):
        return lr * 1 / (1 + 1e-4 * epoch)
    
    callback0 = keras.callbacks.LearningRateScheduler(lr_time_based_decay, verbose=0)
    callback1 = keras.callbacks.EarlyStopping(monitor='loss', patience=50, min_delta=1e-7, verbose=2) #stop criteria
    tf.random.set_seed(111)
    fit_process = model.fit(x_train, y_train, epochs=iterats, verbose = 0, batch_size = 128, shuffle=True,
             callbacks = [callback0,callback1])

    return model, fit_process


