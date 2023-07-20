import tensorflow as tf
from tensorflow.keras import layers

def getPointNet(backbone, classifier, cls_num=3):
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.InputLayer(input_shape=(2048, 3)))

    for filters in backbone:
        net.add(layers.Conv1D(filters=filters, kernel_size=1, use_bias=False))
        net.add(layers.BatchNormalization())
        net.add(layers.ReLU())
    
    net.add(layers.GlobalMaxPooling1D())

    for units in classifier:
        net.add(layers.Dense(units=units, use_bias=False))
        net.add(layers.BatchNormalization())
        net.add(layers.ReLU())
        net.add(layers.Dropout(0.3))
    
    net.add(layers.Dense(units=cls_num, activation='softmax'))

    return net

if __name__ == '__main__':
    pnet = getPointNet(
    backbone=(64,64,64,128,1024),
    classifier=(512, 256)
)

    print(pnet.summary())
