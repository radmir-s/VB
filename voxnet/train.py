import sys
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from tensorflow.keras import callbacks, layers

filters=32
drop=0.4

net = tf.keras.Sequential()
net.add(tf.keras.layers.InputLayer(input_shape=(30, 30, 30, 1)))

net.add(layers.Conv3D(filters=filters, kernel_size=(5, 5, 5), use_bias=False))
net.add(layers.BatchNormalization())
net.add(layers.ReLU())
net.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
net.add(layers.Dropout(drop))

net.add(layers.Conv3D(filters=filters, kernel_size=(3, 3, 3), use_bias=False))
net.add(layers.BatchNormalization())
net.add(layers.ReLU())
net.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
net.add(layers.Dropout(drop))

net.add(layers.Flatten())

net.add(layers.Dense(units=128, use_bias=False))
net.add(layers.BatchNormalization())
net.add(layers.ReLU())
net.add(layers.Dropout(drop))

net.add(layers.Dense(units=3, activation='softmax'))





df = pd.read_csv('data/adni-LR-nodupsY-train-weights.csv')
extra_weight = 0.2

X_train = np.array([np.load(x) for x in df.loc[df.train, 'vox30']])[...,None]
X_train = np.clip(X_train, 0, 2)/2 #clip and normalize
Y_train = df.loc[df.train, 'group'].factorize()[0][...,None]
Y_train = OneHotEncoder().fit_transform(Y_train).toarray()
W_train = (df.loc[df.train, 'weights'].values + extra_weight)/(1+extra_weight)

X_valid = np.array([np.load(x) for x in df.loc[df.valid, 'vox30']])[...,None]
X_valid = np.clip(X_valid, 0, 2)/2 #clip and normalize
Y_valid = df.loc[df.valid, 'group'].factorize()[0][...,None]
Y_valid = OneHotEncoder().fit_transform(Y_valid).toarray()
W_valid = (df.loc[df.valid, 'weights'].values + extra_weight)/(1+extra_weight)

timestamp = datetime.now().strftime("%m.%d.%Y@%H:%M")


adam_opt = tf.keras.optimizers.Adam(learning_rate=lr)
new_model.compile(optimizer=adam_opt,
            weighted_metrics=[],
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
)

reduce_lr = callbacks.ReduceLROnPlateau(
    factor=0.5, 
    min_lr=lr/100, 
    monitor='val_loss', 
    patience=10,
    verbose=0
)

modelstamp = f'./bests/voxmodelnet-t{timestamp}'
csv_log = callbacks.CSVLogger(f'{modelstamp}.log')
checkpoint = callbacks.ModelCheckpoint(
    filepath=modelstamp,
    save_best_only=True,
    verbose=0
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=50,
    verbose=0
)

history = new_model.fit(
    X_train,
    Y_train,
    sample_weight=W_train,
    batch_size=64,
    epochs=epochs,
    shuffle=True,
    validation_data=(X_valid, Y_valid, W_valid),
    callbacks=[checkpoint, reduce_lr, csv_log, early_stop],
    verbose=0,
)
