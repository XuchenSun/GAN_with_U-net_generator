# @author: Xuchen Sun
# @license: General Public License （GPL）
# @contact: xuchens@mun.ca
# @Hardware Note CPU:AMD3900
# @Hardware Note GPU: EVGA GTX1080ti
# @Software Note: Python3.6+CUDA+CUDNN+Pycharm+Tensorflow
# @Version: 1.2
# @Date: 2021-05-10
#import library
import os# set GPU for training
import tensorflow as tf # using in establish U-net Generator and Discriminator
import image_processing
import load_function# load image for training

import Generator_with_Unet# build generator network
import Discriminator# build discriminator net work
import calculation_of_loss_value# calculate loss value




class GAN:
    from matplotlib import pyplot as plt  # draw picture
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'  # enable XLA Devices
    # define the constants
    BUFFER_SIZE = 300  ## change the order of image
    BATCH_SIZE = 1  # batch_size=1 means the trainig is very fast and one picture only trained for 1 time
    IMG_WIDTH = 256  # define the image  size and it must be 256*256*3 because of Unet
    IMG_HEIGHT = 256  # define the image size
    LAMBDA = 98  # define the constant number of Lambada
    OUTPUT_CHANNELS = 3  # channel=1 is enough but 3 can use for color images and it depends on the datasets
    EPOCHS = 200  # set traning epochs and it is limited by the training sets and Max is 213 because of the size in train folder
    gan_count=0
    "Build the Optimizers of generator and discriminator with Adam method and set the teta_1 as 0.5"
    optimizer_of_generator_with_Adam_algorithm = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)  # optimizer the generator
    optimizer_of_discriminator_with_Adam_algorithm = tf.keras.optimizers.Adam(2e-4,
                                                                              beta_1=0.5)  # optimizer the generator

    "Build an instance of class Load_Function_Set()"
    load_function_set = load_function.Load_Function_Set()

    " load data from train folder and test folder"
    image_data_in_train_folder = load_function_set.load_and_prepare_data_from_training_folder()  # load all images and convert them into suitable data for training in training folder
    image_data_in_test_folder = load_function_set.load_and_prepare_data_from_test_folder()  # load all images and convert them into suitable data for training in test folder

    "Build Generator with Unet architecture"
    generator_set = Generator_with_Unet.Generator_Set()
    generator_with_unet = generator_set.Generator_based_on_paper()  # create a generator with U-net



    "Set the type of calculating loss value as Binary"
    value_after_Cross_Entropy_calculation = tf.keras.losses.BinaryCrossentropy(
        from_logits=True)  # use binary method to calculate the cross entropy

    "Build discriminator with VGG architecture"
    discriminator_set = Discriminator.Discriminator_Set()
    discriminator = discriminator_set.Discriminator_based_on_VGG()  # create discriminator by using downsample





    "Set the type of calculating loss value as Categorical"
    value_after_Categorical_Cross_Entropy_calculation = tf.keras.losses.CategoricalCrossentropy  # get the loss value by caterorical cross entropy
    image_processing_set = image_processing.Image_Processing_Set()
    def __init__(self,name,epoch):
        self.name=name
        self.EPOCHS=epoch
        print("Build "+name+" Successfully")
        GAN.gan_count+=1

    def print_summary_of_G_and_D(self):
        self.generator_with_unet.summary()  # print the summary of generator. Total params: 16,667,907,Trainable params: 16,661,635, Non-trainable params: 6,272

        self.discriminator.summary()  # output the information of discriminator

    def plot_model_structure(self):
        tf.keras.utils.plot_model(self.generator_with_unet, show_shapes=True, dpi=64,
                                  to_file='Generator Structure.png')  # plot the model of generator
        tf.keras.utils.plot_model(self.discriminator, show_shapes=True, dpi=64,
                                  to_file='Discriminator Structure.png', )  # plot thte discriminator details
    @tf.function
    def step_of_training_GAN_with_Gradient(self,corresponding_picture, images_of_real_building,
                                           epoch):  # define every step trainning
        with tf.GradientTape() as tape_of_generator, tf.GradientTape() as tape_of_discriminater:  # divided the gradient tap into two parts and this is very important
            output_value_of_generator = self.generator_with_unet(corresponding_picture,
                                                            training=True)  # get the output of generator
            value_of_disc_for_real_image =  self.discriminator([corresponding_picture, images_of_real_building],
                                                         training=True)  # get the disc value of real image
            value_of_disc_from_generated_image =  self.discriminator([corresponding_picture, output_value_of_generator],
                                                               training=True)  # get the disc value of generated image
            gen_total_loss, gen_gan_loss, gen_l1_loss = calculation_of_loss_value.loss_value_of_generator(
                value_of_disc_from_generated_image, output_value_of_generator,
                images_of_real_building)  # get the loss value
            loss_value_of_disc = calculation_of_loss_value.get_the_loss_value_of_discriminator(
                value_of_disc_for_real_image,
                value_of_disc_from_generated_image)  # get the disc loss value
        gradients_value_from_discriminator = tape_of_discriminater.gradient(loss_value_of_disc,
                                                                            self.
                                                                            discriminator.trainable_variables)  # get the gradients value of discriminator
        gradients_value_from_generator = tape_of_generator.gradient(gen_total_loss,
                                                                    self.
                                                                    generator_with_unet.trainable_variables)  # get the gradients value of generator
        self.optimizer_of_generator_with_Adam_algorithm.apply_gradients(zip(gradients_value_from_generator,
                                                                            self.
                                                                            generator_with_unet.trainable_variables))  # optimizer generator by using Adam method
        self.optimizer_of_discriminator_with_Adam_algorithm.apply_gradients(
            zip(gradients_value_from_discriminator,
                self.
                discriminator.trainable_variables))  ##optimizer discriminator Adam method

    def trainning_GAN(self,dataset_of_trainning, dataset_of_test, epochs):
        for epoch in range(epochs):  # define details in every epoch
            for an_corresponding_picture, an_target_image in dataset_of_test.take(
                    1):  # for every image in dataset, there will be a predicted image
                self.image_processing_set.predict_images_from_generator(an_corresponding_picture, an_target_image,
                                                               self.generator_with_unet)  # excuted predict function to generate images
            print("The image of result is saved in current folder after every 2 epochs")
            print("Current epoch number: ", epoch," of ", self.name, " is trainning...........")
            # Train
            for j, (corresponding_picture,
                    images_of_real_building) in dataset_of_trainning.enumerate():  # apply gradient for iteration and change the generator and discriminator

                self.step_of_training_GAN_with_Gradient(corresponding_picture, images_of_real_building,
                                                   epoch)  # excute the details of training in every step

            if epoch % 2 == 0 and epoch > 0:  # save  the image every 2 epoches
                self.plt.savefig(str(epoch)+"Epoch" +self.name+ 'result.jpg')  # first character is the epoch numbers and
            if epoch % 200 == 0 and epoch > 0:  # show image when the program is finished
                # plt.show()# show image
                print("Training is finished")  # hint for training finished

    "Train GAN with the given number of epochs and the data from train folder, test folder "
    def start_train(self):
        self.trainning_GAN( self.image_data_in_train_folder,  self.image_data_in_test_folder,
                  self.EPOCHS)  # Maximize the number of training， but it only has 213 pictures in train sets

