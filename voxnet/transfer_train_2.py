import os
import sys
import pandas as pd
import numpy as np
import argparse

import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
from tensorflow.keras import callbacks, layers

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=1000)
parser.add_argument('-l', '--learning-rate', type=float, default=0.0001)
parser.add_argument('-w', '--weight', action='store_true',
                    help='Set weight to true')
parser.add_argument('-j', '--job-id', type=int)
args = parser.parse_args()

epochs = args.epochs
lr = args.learning_rate
weights = args.weight
jobid = args.job_id

modelnet = tf.keras.models.load_model('./bests/modelnet-t07.26.2023@19:30')

# for layer in modelnet.layers[:1]:
#     layer.trainable = False

df = pd.read_csv('./data/adni-LR-nodupsY-train-weights.csv')
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


adam_opt = tf.keras.optimizers.Adam(learning_rate=lr)
modelnet.compile(optimizer=adam_opt,
            weighted_metrics=[],
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
)

timestamp = datetime.now().strftime("%m.%d.%Y@%H:%M")
print(f"Voxmodelnet: Training started at {timestamp}")

reduce_lr = callbacks.ReduceLROnPlateau(
    factor=0.5, 
    min_lr=lr/100, 
    monitor='val_loss', 
    patience=10,
    verbose=0
)

modelstamp = f'./bests/transfervoxnet-t{timestamp}-j{jobid}'
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

if weights:
    sample_weight=W_train
    validation_data=(X_valid, Y_valid, W_valid)
else:
    sample_weight=None
    validation_data=None   

history = modelnet.fit(
    X_train,
    Y_train,
    sample_weight=sample_weight,
    batch_size=64,
    epochs=epochs,
    shuffle=True,
    validation_data=validation_data,
    callbacks=[checkpoint, reduce_lr, csv_log, early_stop],
    verbose=0,
)


