# @author: Xuchen Sun
# @license: General Public License （GPL）
# @contact: xuchens@mun.ca
# @Hardware Note: CPU:AMD3900
# @Hardware Note: GPU: EVGA GTX1080ti
# @Software Note: Python3.6+CUDA+CUDNN+Pycharm+Pytorch
# @Version: 1.0

import torchvision.models as models_from_torchvision
import torch #use torch package
import torch.nn as torchnn
import Gram_Matrix
import Calculation_Content_Loss
import Calculation_Style_Loss

# judge if support GPU and load to device
running_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load Pre-train VGG19 as model
vgg_net = models_from_torchvision.vgg19(pretrained=True).features # import pre-trained VGG19 with features as CNN model
vgg_net = vgg_net.to(running_device)# using running_device
layers_for_content_loss = ['conv_4']# use conv_4 to get the content loss value
layers_for_style_loss = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']# use conv_1 to conv_5 yo get the style loss value.

# initialize a new CNN model
CNN_Model = torchnn.Sequential()# use sequential to build the CNN model
CNN_Model = CNN_Model.to(running_device)#  using running device

def model_which_get_style_and_content_loss_value(s_image, c_image, cnn=vgg_net, style_weight=1000, content_weight=1,content_layers=layers_for_content_loss,style_layers=layers_for_style_loss):
    #Use the list to store the above six loss functions

    list_of_content_lost = []
    list_of_style_lost = []

    #Style extraction function
    gram_matrix = Gram_Matrix.Gram_Matrix_for_style_features()
    gram_matrix = gram_matrix.to(running_device)

    initialize_the_name_of_layer = 1
    #Traverse vgg19 to find the convolution layer we need
    for filters in cnn:
        # If the layer is nn.Conv2d Object, returns true

        # Otherwise, return false
        if isinstance(filters, torchnn.Conv2d):
            #Add the build-up layer to our model
            filter_names = 'conv_' + str(initialize_the_name_of_layer)
            CNN_Model.add_module(filter_names, filters)

            #Determine whether the build-up layer is used to calculate content loss
            if filter_names in layers_for_content_loss:
                #Here is to put the target into the model and get the target of the layer
                tar_image = CNN_Model(c_image)
                # The target is passed into the specific loss class as a parameter to get a tool function.
                # This function can calculate the content loss of any image and target
                content_loss = Calculation_Content_Loss.Calculation_of_Content_Loss(tar_image, content_weight)
                CNN_Model.add_module('The value of content loss' + str(initialize_the_name_of_layer), content_loss)
                list_of_content_lost.append(content_loss)


            #Similar to content loss, but with one more step: extract style
            if filter_names in layers_for_style_loss:
                tar_image = CNN_Model(s_image)
                tar_image = gram_matrix(tar_image)
                # The target is passed into the specific loss class as a parameter to get a tool function.
                # This function can calculate the style loss of any image and target
                style_loss = Calculation_Style_Loss.Calculation_of_Style_Loss(tar_image, style_weight)
                CNN_Model.add_module('The value of style loss ' + str(initialize_the_name_of_layer), style_loss)
                list_of_style_lost.append(style_loss)# add style loss in the end of list


            initialize_the_name_of_layer += 1
        #For pooling layer and relu layer, we can add them directly
        if isinstance(filters, torchnn.MaxPool2d):
            filter_specific_name = 'pooling_layers' + str(initialize_the_name_of_layer)
            CNN_Model.add_module(filter_specific_name, filters)

        if isinstance(filters, torchnn.ReLU):
            filter_specific_name = 'relu_layers' + str(initialize_the_name_of_layer)
            CNN_Model.add_module(filter_specific_name, filters)
    # To sum up, we have obtained:
    # A specific neural network model,
    # A set of style loss functions (including 5 loss functions of different style objectives)
    # A set of content loss functions (there is only one, you can also define more)
    print("CNN Model Build Successfully")
    print("List Build Successfully")
    return CNN_Model, list_of_style_lost, list_of_content_lost
