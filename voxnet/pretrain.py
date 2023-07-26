import sys
import os
import glob
from datetime import datetime

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import callbacks

from model import getVoxNet2

print('Available physical devices:')
for device in tf.config.list_physical_devices():
    print(device)

epochs = int(sys.argv[1])
data_path = '../modelnet40data/*/30/{}/*.npy'
cls_num = 40

def pipe_data(data_path_frmt, data_split='train', batch_size=64):
    npy_files = glob.glob(data_path_frmt.format(data_split))

    path_ds = tf.data.Dataset.from_tensor_slices(npy_files)
    def load_npy(file_path):
        return np.load(file_path.numpy().decode('utf-8'))[...,None]
    arrays = path_ds.map(lambda x: tf.py_function(func=load_npy, inp=[x], Tout=tf.float32))

    labels = np.array([path.split('/')[-4] for path in npy_files])
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    lookup_layer = tf.keras.layers.experimental.preprocessing.StringLookup()
    lookup_layer.adapt(label_ds)
    integer_labels = label_ds.map(lookup_layer)
    def convert_to_one_hot(label):
        return tf.one_hot(label, depth=cls_num)
    one_hot_labels = integer_labels.map(convert_to_one_hot)

    ds =  tf.data.Dataset.zip((arrays, one_hot_labels))

    ds = ds.cache() 
    ds = ds.shuffle(buffer_size=10000)  
    ds = ds.batch(batch_size)  
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)  

    return ds

train_ds = pipe_data(data_path, 'train')
test_ds = pipe_data(data_path, 'test')

timestamp = datetime.now().strftime("%m.%d.%Y@%H:%M")
print(f"Modelnet: Training started at {timestamp}")

net = getVoxNet2(cls_num)

adam_opt = tf.keras.optimizers.Adam(learning_rate=0.005)
net.compile(optimizer=adam_opt,
            weighted_metrics=[],
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
)

reduce_lr = callbacks.ReduceLROnPlateau(
    factor=0.5, 
    min_lr=0.00001, 
    monitor='val_loss', 
    patience=50,
    verbose=0
)
modelstamp = f'../bests/modelnet-t{timestamp}'
csv_log = callbacks.CSVLogger(f'{modelstamp}.csv')
checkpoint = callbacks.ModelCheckpoint(
    filepath=modelstamp,
    save_best_only=True,
    verbose=0
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=200,
    verbose=0
)

history = net.fit(
    train_ds,
    epochs=epochs,
    validation_data=test_ds,
    callbacks=[checkpoint, reduce_lr, csv_log, early_stop],
    verbose=0,
)

timestamp = datetime.now().strftime("%m.%d.%Y@%H:%M")
print(f"ModelNet: Training ended at {timestamp}")