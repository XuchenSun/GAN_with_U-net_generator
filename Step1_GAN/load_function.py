# @author: Xuchen Sun
# @license: General Public License （GPL）
# @contact: xuchens@mun.ca
# @Hardware Note CPU:AMD3900
# @Hardware Note GPU: EVGA GTX1080ti
# @Software Note: Python3.6+CUDA+CUDNN+Pycharm+Tensorflow
# @Version: 1.2
# @Date: 2021-05-10


import image_processing
import tensorflow as tf # using in establish U-net Generator and Discriminator

#define the constants
BUFFER_SIZE = 404 ## change the order of image
BATCH_SIZE = 1 #  batch_size=1 means the trainig is very fast and one picture only trained for 1 time
IMG_WIDTH = 256 # define the image  size and it must be 256*256*3 because of Unet
IMG_HEIGHT = 256 #define the image size
LAMBDA = 98 #define the constant number of Lambada
OUTPUT_CHANNELS = 3 # channel=1 is enough but 3 can use for color images and it depends on the datasets
EPOCHS = 400 # set traning epochs and it is limited by the training sets
#define function: 1 load   2  random jitter  3  normalize image


"""This class is used for loading data """
class Load_Function_Set():
  image_processing_set = image_processing.Image_Processing_Set()

  def __init__(self):
    print("Build Load Function Set Successfully")

  "Use Functions from image processing set"

  def create_instance_of_image_processing_set(self):
    image_processing_set = image_processing.Image_Processing_Set()
    return image_processing_set

  "Use Functions load image in train_folder"
  def load_image_in_train_folder(self,original_pictures):

    corresponding_picture, images_of_real_building = self.image_processing_set.decode_image(
      original_pictures)  # decode original images
    print("train dataset operation: load image is finished" + '\n')  # show result
    corresponding_picture, images_of_real_building = self.image_processing_set.jitter_prevent_overfitting(
      corresponding_picture, images_of_real_building)  # use  jitter function
    print("train dataset operation:random-jitter image is finished" + '\n')
    corresponding_picture, images_of_real_building = self.image_processing_set.normalize_the_value_of_image(
      corresponding_picture, images_of_real_building)  # use  normalize function to change the value of images
    print("train dataset operation: normalizing image  is finished" + '\n')
    return corresponding_picture, images_of_real_building  # return images

  "Use Functions load image in test_folder"
  def load_image_in_test_folder(self,original_pictures):

    corresponding_picture, images_of_real_building = self.image_processing_set.decode_image(
      original_pictures)  # decode images
    print("test dataset operation: load image is finished" + '\n')
    corresponding_picture, images_of_real_building = self.image_processing_set.adjust_image_size(corresponding_picture,
                                                                                            images_of_real_building,
                                                                                            IMG_HEIGHT,
                                                                                            IMG_WIDTH)  # use  jitter function
    print("test dataset operation: resizing image to 256*256 is finished" + '\n')
    corresponding_picture, images_of_real_building = self.image_processing_set.normalize_the_value_of_image(
      corresponding_picture, images_of_real_building)  # use normalize function to change the function of images
    print("test dataset operation: normalizing image is finished" + '\n')
    return corresponding_picture, images_of_real_building  # return the poictures

  "Use Functions load and prepare the data"
  def load_and_prepare_data_from_training_folder(self):

    self.image_processing_set.test_for_decoding_image_function()  # decoding image
    image_data_in_train_folder = tf.data.Dataset.list_files('train/*.jpg')  # load train pictures
    image_data_in_train_folder = image_data_in_train_folder.map(self.load_image_in_train_folder,
                                                                num_parallel_calls=tf.data.AUTOTUNE)  # parallel loading
    image_data_in_train_folder = image_data_in_train_folder.shuffle(BUFFER_SIZE)  # BUFFER_SIZE=300
    image_data_in_train_folder = image_data_in_train_folder.batch(BATCH_SIZE)  # BATCH_SIZE=1
    print("Loading and preparing data from train folder is finished")
    return image_data_in_train_folder  # return the data in train folder

  def load_and_prepare_data_from_test_folder(self):

    image_data_in_test_folder = tf.data.Dataset.list_files('test/*.jpg')  # load pictures in test folder
    image_data_in_test_folder = image_data_in_test_folder.map(self.load_image_in_test_folder)  # map data set
    image_data_in_test_folder = image_data_in_test_folder.batch(BATCH_SIZE)  # BATCH_SIZE=1
    print("input image size is 256*256")  # print hint
    print("After using Cov2D, the output image size is 256*256")  # print hint
    print("loading data from test folder is finished")  # print hint
    return image_data_in_test_folder  # return the data in test folder