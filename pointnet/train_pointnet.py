import sys
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from tensorflow.keras import layers, callbacks

from model import getPointNet

print('Available physical devices:')
for device in tf.config.list_physical_devices():
    print(device)

if len(sys.argv) != 5:
    print("Usage: python train_pointnet.py $epochs $backbone $classifier $jit")
    sys.exit(1)

epochs = int(sys.argv[1])
backbone = [int(x) for x in sys.argv[2].split('.')]
classifier = [int(x) for x in sys.argv[3].split('.')]
lr = float(sys.argv[4])

df = pd.read_csv('../data/adni-LR-nodupsY-train-weights.csv')
extra_weight = 0.2

X_train = np.array([np.load(x.replace('data/','../data/',1)) for x in df.loc[df.train, 'npy-2048']])
Y_train = df.loc[df.train, 'group'].factorize()[0][...,None]
Y_train = OneHotEncoder().fit_transform(Y_train).toarray()
W_train = (df.loc[df.train, 'weights'].values + extra_weight)/(1+extra_weight)

X_valid = np.array([np.load(x.replace('data/','../data/',1)) for x in df.loc[df.valid, 'npy-2048']])
Y_valid = df.loc[df.valid, 'group'].factorize()[0][...,None]
Y_valid = OneHotEncoder().fit_transform(Y_valid).toarray()
W_valid = (df.loc[df.valid, 'weights'].values + extra_weight)/(1+extra_weight)

timestamp = datetime.now().strftime("%m.%d.%Y@%H:%M")
print(f"Point: Training started at {timestamp}")

net = getPointNet(
    backbone=backbone,
    classifier=classifier
)

adam_opt = tf.keras.optimizers.Adam(learning_rate=lr)
net.compile(optimizer=adam_opt,
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

modelstamp = f'voxnet-b{sys.argv[2]}-c{sys.argv[3]}-t{timestamp}'
csv_log = callbacks.CSVLogger(f'{modelstamp}.csv')
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

history = net.fit(
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

timestamp = datetime.now().strftime("%m.%d.%Y@%H:%M")
print(f"Point: Training ended at {timestamp}")