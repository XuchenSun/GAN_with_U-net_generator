# @author: Xuchen Sun
# @license: General Public License （GPL）
# @contact: xuchens@mun.ca
# @Hardware Note CPU:AMD3900
# @Hardware Note GPU: EVGA GTX1080ti
# @Software Note: Python3.6+CUDA+CUDNN+Pycharm+Tensorflow
# @Version: 1.2
# @Date: 2021-05-10
import tensorflow as tf # using in establish U-net Generator and Discriminator

"""This function is used for building generator with Unet architecture."""
def build_upsample_layers(filters, size, apply_dropout=False):#build upsample layers

    initializer = tf.random_normal_initializer(0., 0.019)# initializer with random 0.019
    upsample_layers = tf.keras.Sequential()#use sequential to build layers
    upsample_layers.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,padding='same',kernel_initializer=initializer,use_bias=False))
    upsample_layers.add(tf.keras.layers.BatchNormalization())#use batchNormalization to build layer
    if apply_dropout:
        upsample_layers.add(tf.keras.layers.Dropout(0.51))#add dropout
    upsample_layers.add(tf.keras.layers.ReLU())#add relu layer
    return upsample_layers

"""This function is used for building generator with Unet architecture and discriminator"""
def build_downsample_layers(filters, size, apply_batchnorm=True):#build downsample layers
    initializer = tf.random_normal_initializer(0., 0.019)# initializer with random 0.019
    downsample_layers = tf.keras.Sequential()# use sequential method to build layers
    downsample_layers.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        downsample_layers.add(tf.keras.layers.BatchNormalization())#add batch layers
    downsample_layers.add(tf.keras.layers.LeakyReLU())#add leakyRelu layer
    return downsample_layers

