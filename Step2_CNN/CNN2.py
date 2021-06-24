# @author: Xuchen Sun
# @license: General Public License （GPL）
# @contact: xuchens@mun.ca
# @Hardware Note: CPU:AMD3900
# @Hardware Note: GPU: EVGA GTX1080ti
# @Software Note: Python3.6+CUDA+CUDNN+Pycharm+Pytorch
# @Version: 1.0
import pandas as pd # use functions in panda package to save the loss value into html files
import torch #use torch package
import torch.nn as torchnn
import torchvision.transforms as transforms_from_torchvision
import torchvision.models as models_from_torchvision
import L_BFGS
import Gram_Matrix
import loading_image_to_tensor
import Calculation_Content_Loss
import Calculation_Style_Loss
import Models_get_the_loss_value
from torch.autograd import Variable as autograd_from_torch
#set the size of image
SIZE_IMAGE = 512
#Set EPOCH times
EPOCH=120# 120 times is enough to get a good image for game development
list_for_content_lost = []#build list for content lost
list_for_style_lost = []#build list for style_lost
vgg_net = models_from_torchvision.VGG# use vgg net
# judge if support GPU and load to device
running_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# pytorch use GPU for training
# load Pre-train VGG19 as model
vgg_net = models_from_torchvision.vgg19(pretrained=True).features# pretrained=True means this models import pretrainning networks
vgg_net = vgg_net.to(running_device)# add running device
layers_for_content_loss = ['conv_4']#use conv_4 as getting content loss value
layers_for_style_loss = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']#use 5 layers to get the loss value of style
#load images
#load_c_image = "content.jpg"  # load contenct image
#load_c_image="content2.png" # load contenct2 image #manually change one by one
load_c_image="content3.png" # load contenct3 image #manually change one by one
load_s_image = "style.png"# load stlye image
# load style image to tensor
style_img = loading_image_to_tensor.load_img_to_tensor(load_s_image)# the image must be loaded to tensor for ML
# change img to Variable object，make it to calculate value
style_img = autograd_from_torch(style_img).to(running_device)# auto grad
# load content image
content_img = loading_image_to_tensor.load_img_to_tensor(load_c_image)# change the image to tensor
content_img = autograd_from_torch(content_img).to(running_device)# auto grad
print(style_img.size(), content_img.size())# check if correctly load and print the size
# define the test of loss function
value_of_contect_loss = Calculation_Content_Loss.Calculation_of_Content_Loss(content_img, 1)# use function in other file to calculate the content loss value
# random picture
rand_img = torch.randn(content_img.data.size(), device=running_device)# prevent over fitting
value_of_contect_loss.forward(rand_img)# calculation of moving forward for rand_image
print("Test for content loss is sucessful")
print("Test value is "+str(value_of_contect_loss.loss))# check if the loss value can be calculated correctly
#define Gram Matrix
gram_matrix = Gram_Matrix.Gram_Matrix_for_style_features()# import class Gram_Matrix to calculate style features
target_value_after_gram_matrix = gram_matrix(style_img)# calculation the value of style image
# because style_img size is 3 Therefore, the style features are as follows 3×3

#Parameters required for incoming model
style_loss_value = Calculation_Style_Loss.Calculation_of_Style_Loss(target_value_after_gram_matrix, 1000)#import class Calculation_Style_Loss to calculate the style loss
#Pass in a random picture to test
random_img = torch.randn(style_img.data.size(), device=running_device)#transfer the value of style loss
#The loss function layer propagates forward to get the loss function
style_loss_value.forward(random_img)# moving forward calculation
style_loss_value.loss# figure out the loss value
# initialize a new CNN model
CNN_Model = torchnn.Sequential()# using sequential type to build CNN model
CNN_Model = CNN_Model.to(running_device)# add model to device
#Construct the network model and return these loss functions

architecture = CNN_Model.cuda()# define the architecture of CNN model
print(architecture)# print architecture
CNN_Model, list_of_style_lost, list_of_content_lost = Models_get_the_loss_value.model_which_get_style_and_content_loss_value(style_img, content_img)# calculate the loss value

#Input a random picture to test
L_BFGS.optimier_with_L_BFGS_Algorithms(random_img) #initialize optimier with L_BFGS algorithems

#Incoming input_ IMG is the value of each pixel in G, which can be a random picture
# transfer style
def transfer_stlye_function(content_img, style_img, input_img, num_epoches):
    print('Start to build the Neural style transfer model..')
    #Specify the parameters to be optimized. Here, input_ Param is the value of each pixel in G
    input_param, optimizer_L_BFGS = L_BFGS.optimier_with_L_BFGS_Algorithms(input_img)

    print('Neural style transfer program is initialising...')
    epoch = [0]# initialize eopoch
    while epoch[0] < num_epoches:#loop trainning
        #Here we define the calculation method of total loss
        def closure():# define close step
            input_param.data.clamp_(0, 1)  #Update image data
            #At this time, G is passed into the model to get the output of each network layer
            CNN_Model(input_param)
            loss_value_of_style = 0 # initialize the loss value of style
            loss_value_of_content = 0#initialize the loss value of content
            #Gradient before emptying
            optimizer_L_BFGS.zero_grad()
            #The total loss is calculated and the gradient of each loss is obtained
            for style_loss_value in list_of_style_lost:# define loop for calculation
                loss_value_of_style += style_loss_value.backward()# using moving backward calculation for the loss value of style
            for content_loss_value in list_of_content_lost:# define loop for calculation
                loss_value_of_content  += content_loss_value.backward()# using moving backward calculation for the loss value of content

            epoch[0] += 1 #set epoch++
            # Here, every iteration is output once

            if epoch[0] % 1 == 0:#set print log for every loop
                print('Neural style transfer program(CNN) is trainning {}/100'.format(epoch))
                print('The value of style Loss is : {:.4f} The value of content loss is : {:.4f}'.format(loss_value_of_style.data.item(), loss_value_of_content .data.item()))
                list_for_content_lost.append(format(loss_value_of_content .data.item()))# load loss value to a list
                list_for_style_lost.append(format(loss_value_of_style.data.item()))# load loss value to a list
                print("Running.......")

            return loss_value_of_style + loss_value_of_content# Returns the sum of two loss values
        #Update G
        optimizer_L_BFGS.step(closure)# for every step, use optimizer to change and this is the key in iteration
    #Return to G after training, g at this time
    return input_param.data#return param data
#Initialize image G
input_img = content_img.clone()# the input image is the clone type of content image, so the content loss is smaller at first training
#Model training, and return to the picture
def training_and_get_the_data(c_image,s_image,input_img,EPOCH):
    output_data_of_image=(c_image,s_image,input_img,EPOCH)
    # The picture can be converted into PIL type for easy display
    #output_image = transforms.ToPILImage()(output_data_of_image.cpu().squeeze(0))
    #output_image.save("final.jpg")
    #print("finish training and the final picture is final.jpg")
training_and_get_the_data(content_img, style_img, input_img,EPOCH)# execute the training
output_data_of_image = transfer_stlye_function(content_img, style_img, input_img, num_epoches=EPOCH)# get the output image G
#The picture can be converted into PIL type for easy display
output_image = transforms_from_torchvision.ToPILImage()(output_data_of_image.cpu().squeeze(0))# transfer data to image
output_image.save("final3.jpg")# output the image
dataframe_of_content_loss_value=pd.DataFrame(list_for_content_lost)# transfer data of content loss value from list type to dataframe type
dataframe_of_style_loss_value=pd.DataFrame(list_for_style_lost)# transfer data of style loss value from list type to dataframe type
dataframe_of_content_loss_value.to_html('content_loss_value.html')# output the data of content loss value in the type of html
dataframe_of_style_loss_value.to_html('style_loss_value.html')# output the data of style loss value in the type of html
print("Traning Times: "+str(EPOCH))#output the epoch tiem
print("Training is finished and the final picture is final.jpg")# print helpful information
print("Training is finished and the loss values are in the html files of content_loss_value and style_loss_value ")#print helpful information
