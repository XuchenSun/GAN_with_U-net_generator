# @author: Xuchen Sun
# @license: General Public License （GPL）
# @contact: xuchens@mun.ca
# @Hardware Note CPU:AMD3900
# @Hardware Note GPU: EVGA GTX1080ti
# @Software Note: Python3.6+CUDA+CUDNN+Pycharm+Tensorflow
# @Version: 1.2
# @Date: 2021-05-10
import tensorflow as tf # using in establish U-net Generator and Discriminator
from matplotlib import pyplot as plt# draw picture


#define the constants
BUFFER_SIZE = 404 ## change the order of image
BATCH_SIZE = 1 #  batch_size=1 means the trainig is very fast and one picture only trained for 1 time
IMG_WIDTH = 256 # define the image  size and it must be 256*256*3 because of Unet
IMG_HEIGHT = 256 #define the image size
LAMBDA = 98 #define the constant number of Lambada
OUTPUT_CHANNELS = 3 # channel=1 is enough but 3 can use for color images and it depends on the datasets
EPOCHS = 400 # set traning epochs and it is limited by the training sets


"""This Class is used for build Image Process Set"""
class Image_Processing_Set:
    def __init__(self):
        print("..")


    """This function is used for decoding images into values in 3D matrix """
    def decode_image(self,original_pictures):  # Convert a picture into a 3D matrix and 3D in here means RGB
        image_after_reading = tf.io.read_file(original_pictures)  # read images data
        image_after_decode = tf.image.decode_jpeg(image_after_reading)  # decode jpeg type data
        # because the image from dataset is a combination so it must be split two two images in half
        width_in_tf = tf.shape(image_after_decode)[1]  # get the shape of pictures
        width_in_half = width_in_tf // 2  # because the two images combined together, it is necessary to divide them
        data_of_real_image_after_decoding = image_after_decode[:, :width_in_half, :]  # get the real image of building
        input_image_after_decode = image_after_decode[:, width_in_half:, :]  # get the input image
        input_image_float32 = tf.cast(input_image_after_decode, tf.float32)  # get the data from float32 type
        Mask_With_OPC_image_float32 = tf.cast(data_of_real_image_after_decoding,
                                                  tf.float32)  # get the data from float32 type
        return input_image_float32, Mask_With_OPC_image_float32  # return the data

    """This function is used for testing if the decoding can work"""

    def test_for_decoding_image_function(self):
        corresponding_picture, images_of_Mask_With_OPC = self.decode_image('val/11.jpg')
            # casting to int for matplotlib to show the image
        plt.figure("Input Image")
        plt.imshow(corresponding_picture / 255.0)
        plt.figure("Real Building Image")
        plt.imshow(images_of_Mask_With_OPC / 255.0)
        print("First image should be input image of houses between the value of 0 to 255")
        print("Second image should be real image of houses between the value of 0 to 255")

    """This function is used for changing image size"""

    def adjust_image_size(self,corresponding_picture, images_of_Mask_With_OPC, height_of_image, width_of_image):
        corresponding_picture = tf.image.resize(corresponding_picture, [height_of_image, width_of_image],
                                                    method=tf.image.ResizeMethod.AREA)  # use area method to resize the input image
        images_of_Mask_With_OPC = tf.image.resize(images_of_Mask_With_OPC, [height_of_image, width_of_image],
                                                      method=tf.image.ResizeMethod.AREA)  # use area method to resize the real image
        return corresponding_picture, images_of_Mask_With_OPC  # return the two image after resize

    """This function is used for prevent overfitting of training"""
    def prevent_overfitting(self,corresponding_picture, images_of_Mask_With_OPC):
        image_in_stack = tf.stack([corresponding_picture, images_of_Mask_With_OPC], axis=0)  # save data in stack
        image_after_cropping = tf.image.random_crop(image_in_stack,
                                                        size=[2, IMG_HEIGHT, IMG_WIDTH, 3])  # crop image in stack
        return image_after_cropping[0], image_after_cropping[1], image_after_cropping[1]  # return images

    """This function is used for normalizing the images to [-1, 1]"""
    def normalize_the_value_of_image(self,corresponding_picture, Mask_With_OPC):
        corresponding_picture = (
                                                corresponding_picture / 127.5) - 1  # normalize the value of corresponding pictures between[-1,1]
        images_of_Mask_With_OPC = (Mask_With_OPC / 127.5) - 1  # nor
        return corresponding_picture, images_of_Mask_With_OPC

    """This function is used for normalizing the images to [-1, 1]"""
    @tf.function()
    def jitter_prevent_overfitting(self,corresponding_picture,
                                       Mask_With_OPC):  # define jitter function to prevent overfitting
        corresponding_picture, images_of_Mask_With_OPC = self.adjust_image_size(corresponding_picture, Mask_With_OPC, 264,
                                                                               264)  # randomly cropping to 264 x 264 x 3
        corresponding_picture, images_of_Mask_With_OPC, images_of_real_building2 = self.prevent_overfitting(
                corresponding_picture, images_of_Mask_With_OPC)  # data images are processed to prevent overfitting
        if tf.random.uniform(()) > 0.4:
                # random mirroring
                corresponding_picture = tf.image.flip_left_right(corresponding_picture)  # flip function
                images_of_Mask_With_OPC = tf.image.flip_left_right(images_of_Mask_With_OPC)  # flip function
        return corresponding_picture, images_of_Mask_With_OPC  # return two pictures

    """This function is used for the generating new images by using trained-generator."""

    def predict_images_from_generator(self,picture_without_opc, Mask_With_OPC, Generater_layers):
        predicted_image_from_generater = Generater_layers(picture_without_opc, training=True)  # generaterd image
        plt.figure(figsize=(35, 35))  # define the size of image
        display_list = [picture_without_opc[0], Mask_With_OPC[0],
                            predicted_image_from_generater[0]]  # define a list to store the image
        name_of_images = ['Mask_Without_OPC', 'Mask_With_OPC', 'GAN predict Mask_With_OPC']  # define the image title
        for k in range(3):  # show image
            plt.subplot(1, 3, k + 1)  # show image
            plt.title(name_of_images[k])  # set the title
            # getting the pixel values between [0, 1] to plot it to screen.
            plt.imshow(display_list[k] * 0.5 + 0.5)  # show image
            plt.axis('off')  # prevent error