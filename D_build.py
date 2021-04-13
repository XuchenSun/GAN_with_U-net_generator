import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential,models
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'# enable training of GPU
tf.random.set_seed(5483)# define random iniation


#define conv layer of discrimator


conv_layers = [

    #2Cov and 1 pool

    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,strides=(1, 1)),
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,strides=(2, 2)),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    #second part
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,strides=(1, 1)),
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,strides=(2, 2)),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    #Third part
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,strides=(1, 1)),
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,strides=(2, 2)),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    #Forth part
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,strides=(1, 1)),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,strides=(2, 2)),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    #Five part
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,strides=(1, 1)),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,strides=(2, 2)),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

]
discrimator_conv_net = Sequential(conv_layers)

#define full connection layer of discrimator
discrimator_FCN = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(100, activation=None),
    ])

discrimator_conv_net.build(input_shape=[None, 32, 32, 3])
discrimator_FCN.build(input_shape=[None, 512])
discrimator_conv_net.summary()
discrimator_FCN.summary()






