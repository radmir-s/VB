import tensorflow as tf
from tensorflow.keras import layers

def getVoxNet(backbone, classifier, cls_num=3, activ='relu'):
    filt1, filt2 = backbone
    
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.InputLayer(input_shape=(48, 32, 24, 1)))

    net.add(layers.Conv3D(filters=filt1, kernel_size=(5, 5, 5), use_bias=False))
    net.add(layers.BatchNormalization())
    net.add(layers.ReLU())
    net.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    
    net.add(layers.Conv3D(filters=filt2, kernel_size=(3, 3, 3), use_bias=False))
    net.add(layers.BatchNormalization())
    net.add(layers.ReLU())
    net.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    net.add(layers.Flatten())

    for units in classifier:
        net.add(layers.Dense(units=units, use_bias=False))
        net.add(layers.BatchNormalization())
        net.add(layers.ReLU())
        net.add(layers.Dropout(0.3))
    
    net.add(layers.Dense(units=cls_num, activation='softmax'))

    return net

if __name__ == '__main__':
    vnet = getVoxNet(
        backbone=(32,32),
        classifier=(128,32)
    )

    print(vnet.summary())
