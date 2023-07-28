import os
import sys
import pandas as pd
import numpy as np

import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
from tensorflow.keras import callbacks, layers


epochs = int(sys.argv[1])
class_layers = map(int, sys.argv[2].split('.'))
lr = float(sys.argv[3])

modelnet = tf.keras.models.load_model('./bests/modelnet-t07.26.2023@19:30')

print('Modelnet summary:')
print(modelnet.summary())

inputs = tf.keras.Input(shape=modelnet.input_shape[1:])
x = inputs

for layer in modelnet.layers[:-5]:
    if not isinstance(layer, layers.Dropout):
        x = layer(x)

for u in class_layers:
    x = layers.Dense(units=u, activation='softmax')(x)
    x = layers.Dropout(0.3)(x)

x = layers.Dense(units=3, activation='softmax')(x)

outputs = x

new_model = tf.keras.Model(inputs, outputs)

print('New model summary:')
print(new_model.summary())

for layer in new_model.layers[:-1]:
    layer.trainable = False

df = pd.read_csv('./data/adni-LR-nodupsY-train-weights.csv')
extra_weight = 0.2

X_train = np.array([np.load(x) for x in df.loc[df.train, 'vox30']])[...,None]
X_train = np.clip(X_train, 0, 10)/10 #clip and normalize
Y_train = df.loc[df.train, 'group'].factorize()[0][...,None]
Y_train = OneHotEncoder().fit_transform(Y_train).toarray()
W_train = (df.loc[df.train, 'weights'].values + extra_weight)/(1+extra_weight)

X_valid = np.array([np.load(x) for x in df.loc[df.valid, 'vox30']])[...,None]
X_valid = np.clip(X_valid, 0, 10)/10 #clip and normalize
Y_valid = df.loc[df.valid, 'group'].factorize()[0][...,None]
Y_valid = OneHotEncoder().fit_transform(Y_valid).toarray()
W_valid = (df.loc[df.valid, 'weights'].values + extra_weight)/(1+extra_weight)


adam_opt = tf.keras.optimizers.Adam(learning_rate=lr)
new_model.compile(optimizer=adam_opt,
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


