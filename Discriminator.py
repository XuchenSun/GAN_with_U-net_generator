# @author: Xuchen Sun
# @license: General Public License （GPL）
# @contact: xuchens@mun.ca
# @Hardware Note CPU:AMD3900
# @Hardware Note GPU: EVGA GTX1080ti
# @Software Note: Python3.6+CUDA+CUDNN+Pycharm+Tensorflow
# @Version: 1.2
# @Date: 2021-05-10
import tensorflow as tf # using in establish U-net Generator and Discriminator



"""This class is used for building Discriminator."""



class Discriminator_Set:
    def __init__(self):
        print("Build Discriminator Set Successfully")

    def Discriminator_based_on_VGG(self):
        initializer = tf.random_normal_initializer(0., 0.009)  # random normal value to initializer

        "define two input layers"
        input_image_layer = tf.keras.layers.Input(shape=[256, 256, 3],
                                                  name='input_image')  # set a layer to get the input image
        target_image_layer = tf.keras.layers.Input(shape=[256, 256, 3],
                                                   name='target_image')  # set a layer to get the target image
        layer0 = tf.keras.layers.concatenate(
            [input_image_layer, target_image_layer])  # combine two layers into one level

        "define repeat2-1 layer of Discriminator "
        repeat2_1 = tf.keras.Sequential()
        repeat2_1.add(tf.keras.layers.Conv2D(64, 3, strides=1, kernel_initializer=initializer, use_bias=False))
        repeat2_1.add(tf.keras.layers.LeakyReLU())
        repeat2_1 = repeat2_1(layer0)

        "define conv-1 layer  of Discriminator"
        conv_1 = tf.keras.Sequential()
        conv_1.add(tf.keras.layers.Conv2D(64, 3, strides=2, kernel_initializer=initializer, use_bias=False))
        conv_1.add(tf.keras.layers.LeakyReLU())
        conv_1 = conv_1(repeat2_1)

        "define repeat2-2 layer of Discriminator"
        repeat2_2 = tf.keras.Sequential()
        repeat2_2.add(tf.keras.layers.Conv2D(128, 3, strides=1, kernel_initializer=initializer, use_bias=False))
        repeat2_2.add(tf.keras.layers.LeakyReLU())
        repeat2_2 = repeat2_2(conv_1)

        "define conv-2 layer  of Discriminator "
        conv_2 = tf.keras.Sequential()
        conv_2.add(tf.keras.layers.Conv2D(128, 3, strides=2, kernel_initializer=initializer, use_bias=False))
        conv_2.add(tf.keras.layers.LeakyReLU())
        conv_2 = conv_2(repeat2_2)

        "define repeat3_1 layer of Discriminator"
        repeat3_1 = tf.keras.Sequential()
        repeat3_1.add(tf.keras.layers.Conv2D(256, 3, strides=1, kernel_initializer=initializer, use_bias=False))
        repeat3_1.add(tf.keras.layers.LeakyReLU())
        repeat3_1 = repeat3_1(conv_2)

        "define conv-3 layer  of Discriminator "
        conv_3 = tf.keras.Sequential()
        conv_3.add(tf.keras.layers.Conv2D(256, 3, strides=2, kernel_initializer=initializer, use_bias=False))
        conv_3.add(tf.keras.layers.LeakyReLU())
        conv_3 = conv_3(repeat3_1)

        "define repeat3_2 of Discriminator "
        repeat3_2 = tf.keras.Sequential()
        repeat3_2.add(tf.keras.layers.Conv2D(512, 3, strides=1, kernel_initializer=initializer, use_bias=False))
        repeat3_2.add(tf.keras.layers.LeakyReLU())
        repeat3_2 = repeat3_2(conv_3)

        "define conv-4 layer  of Discriminator"
        conv_4 = tf.keras.Sequential()
        conv_4.add(tf.keras.layers.Conv2D(512, 3, strides=2, kernel_initializer=initializer, use_bias=False))
        conv_4.add(tf.keras.layers.LeakyReLU())
        conv_4 = conv_4(repeat3_2)

        "define repeat3_3 layer of Discriminator"
        repeat3_3 = tf.keras.Sequential()
        repeat3_3.add(tf.keras.layers.Conv2D(512, 3, strides=1, kernel_initializer=initializer, use_bias=False))
        repeat3_3.add(tf.keras.layers.LeakyReLU())
        repeat3_3 = repeat3_3(conv_4)

        "define conv-5 layer of Discriminator"
        conv_5 = tf.keras.Sequential()
        conv_5.add(tf.keras.layers.Conv2D(512, 3, strides=2, kernel_initializer=initializer, use_bias=False))
        conv_5.add(tf.keras.layers.LeakyReLU())
        conv_5 = conv_5(repeat3_3)

        "define fc-1 of Discriminator"
        fc_1 = tf.keras.Sequential()
        fc_1.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
        fc_1 = fc_1(conv_5)

        "define fc-2  of Discriminator"
        fc_2 = tf.keras.Sequential()
        fc_2.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
        fc_2 = fc_2(fc_1)

        "define fc-3  of Discriminator"
        fc_3 = tf.keras.Sequential()
        fc_3.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
        fc_3 = fc_3(fc_2)

        return tf.keras.Model(inputs=[input_image_layer, target_image_layer], outputs=fc_3)  # return the model

