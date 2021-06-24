# @author: Xuchen Sun
# @license: General Public License （GPL）
# @contact: xuchens@mun.ca
# @Hardware Note CPU:AMD3900
# @Hardware Note GPU: EVGA GTX1080ti
# @Software Note: Python3.6+CUDA+CUDNN+Pycharm+Tensorflow
# @Version: 1.2
# @Date: 2021-05-10
import tensorflow as tf # using in establish U-net Generator and Discriminator
import build_layers








#define the constants
BUFFER_SIZE = 404 ## change the order of image
BATCH_SIZE = 1 #  batch_size=1 means the trainig is very fast and one picture only trained for 1 time
IMG_WIDTH = 256 # define the image  size and it must be 256*256*3 because of Unet
IMG_HEIGHT = 256 #define the image size
LAMBDA = 98 #define the constant number of Lambada
OUTPUT_CHANNELS = 3 # channel=1 is enough but 3 can use for color images and it depends on the datasets
EPOCHS = 200 # set traning epochs and it is limited by the training sets

## Build the Generator(U-net type)

# Use U-Net architecture as generator

# Every data block in the encoder go through
# 1Conv 2Batchnorm 3Leaky ReLU

# Every data block in the decoder is go through 1Transposed Conv 2Batchnorm 3Dropout 4ReLU

# Attention:there is skip connections between the encoder and decoder(this is why they are different).
"""This Class is used for building Generator network with Unet type."""

class Generator_Set:
    def __init__(self):
        print("Build Generator Set Successfully")

    ## Build the Generator(U-net type)

    # Use U-Net architecture as generator

    # Every data block in the encoder go through
    # 1Conv 2Batchnorm 3Leaky ReLU

    # Every data block in the decoder is go through 1Transposed Conv 2Batchnorm 3Dropout 4ReLU

    # Attention:there is skip connections between the encoder and decoder(this is why they are different).
    """This function is used for building Generator network with Unet type."""

    def Generator_based_on_paper(self):
        initializer = tf.random_normal_initializer(0., 0.009)
        layer0 = tf.keras.layers.Input(shape=[256, 256, 3])  # define the input layer

        downsample_layers = [

            # build the layer conv-1
            build_layers.build_downsample_layers(16, 5, apply_batchnorm=False),

            # build the layer conv-2
            build_layers.build_downsample_layers(64, 5),

            # build the layer conv-3
            build_layers.build_downsample_layers(128, 5),

            # build the layer conv-4
            build_layers.build_downsample_layers(512, 5),

            # build the layer conv-5
            build_layers.build_downsample_layers(1024, 5),  # define specific layers
        ]  # define specific layers
        print("Generator:Downsample Filters Build Successfully!")  # print hint if the downsample is finished
        upsample_layers = [

            # SPSR layers need more work
            # build SPSR-5 layer
            build_layers.build_upsample_layers(2048, 3, apply_dropout=True),  # SPSR layer
            # build SPSR-4 layer
            build_layers.build_upsample_layers(512, 3, apply_dropout=True),  # SPSR
            # build SPSR-3 layer
            build_layers.build_upsample_layers(256, 3, apply_dropout=True),  # SPSR

            # build SPSR-2 layer
            build_layers.build_upsample_layers(64, 3, apply_dropout=True),  # SPSR
            # build SPSR-1 layer
            build_layers.build_upsample_layers(4, 3, apply_dropout=True),  # SPSR

        ]
        print("Generator:Upsample Filters Build Successfully!")

        generator_network = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same',
                                                            kernel_initializer=initializer,
                                                            activation='tanh')  # define layers from keras
        generator_model = layer0  # define generator model
        # build skip connects between encoder and decoder
        # Downsampling through the model
        skip_connections = []  # define skip connections to transfer information
        for layers in downsample_layers:
            generator_model = layers(generator_model)  # define layers
            skip_connections.append(generator_model)  # build a connection between two layers
        skip_connections = reversed(skip_connections[:-1])  # transfer informations
        # Upsampling and establishing the skip connections
        for filters, skip in zip(upsample_layers, skip_connections):  # use zip to accelarate
            generator_model = filters(generator_model)  # generater model
            generator_model = tf.keras.layers.Concatenate()(
                [generator_model, skip])  # use concatenate command to generate model
        generator_model = generator_network(generator_model)  # use generator model
        print("Generator: Skip Connection Build Successfully!")  # print hint
        print("Generator Build Successfully!")  # print hint

        return tf.keras.Model(inputs=layer0, outputs=generator_model)  # return Generator Network
