# @author: Xuchen Sun
# @license: General Public License （GPL）
# @contact: xuchens@mun.ca
# @Hardware Note CPU:AMD3900
# @Hardware Note GPU: EVGA GTX1080ti
# @Software Note: Python3.6+CUDA+CUDNN+Pycharm+Tensorflow
# @Version: 1.2
# @Date: 2021-05-10
import tensorflow as tf # using in establish U-net Generator and Discriminator


#define the constants
BUFFER_SIZE = 404 ## change the order of image
BATCH_SIZE = 1 #  batch_size=1 means the trainig is very fast and one picture only trained for 1 time
IMG_WIDTH = 256 # define the image  size and it must be 256*256*3 because of Unet
IMG_HEIGHT = 256 #define the image size
LAMBDA = 98 #define the constant number of Lambada
OUTPUT_CHANNELS = 3 # channel=1 is enough but 3 can use for color images and it depends on the datasets
EPOCHS = 213 # set traning epochs and it is limited by the training sets

loss_value=[]
value_after_Cross_Entropy_calculation = tf.keras.losses.BinaryCrossentropy(from_logits=True)#use Binary Cross Entropy to calculate loss value
value_after_Categorical_Cross_Entropy_calculation=tf.keras.losses.CategoricalCrossentropy(from_logits=True)# use Categorival Cross Entropy to calculate loss value
"""This function is used for the calculation of loss value in generator."""
def loss_value_of_generator(disc_value_from_generated, output_from_Gan, target_value):
  loss_value_of_generated = value_after_Cross_Entropy_calculation(tf.ones_like(disc_value_from_generated), disc_value_from_generated)# Binary Cross Entropy method

  loss_value_of_mean = tf.reduce_mean(tf.abs(target_value - output_from_Gan))# use mean method to get the value
  loss_value_of_all = loss_value_of_generated + (loss_value_of_mean*LAMBDA )#LAMBDA = 98 and it is used in the calculation of loss value in all
  return loss_value_of_all, loss_value_of_generated, loss_value_of_mean# return the loss value
"""This function is used for the calculation of loss value in Discriminator."""
def get_the_loss_value_of_discriminator(disc_value_from_real_image, disc_value_from_generator):
  loss_value_of_real_image = value_after_Cross_Entropy_calculation(tf.ones_like(disc_value_from_real_image), disc_value_from_real_image)# Binary Cross Entropy method
  loss_value_of_input_image=value_after_Categorical_Cross_Entropy_calculation(tf.ones_like(disc_value_from_real_image), disc_value_from_real_image)#use Categorival Cross Entropy to calculate loss value
  loss_value_of_generated_image = value_after_Cross_Entropy_calculation(tf.zeros_like(disc_value_from_generator), disc_value_from_generator)# Binary Cross Entropy method
  loss_disc_value_of_all = loss_value_of_real_image + loss_value_of_generated_image#  get the all loss value
  loss_value=loss_value_of_input_image
  return loss_disc_value_of_all# return the all loss value
