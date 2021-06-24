# @author: Xuchen Sun
# @license: General Public License （GPL）
# @contact: xuchens@mun.ca
# @Hardware Note: CPU:AMD3900
# @Hardware Note: GPU: EVGA GTX1080ti
# @Software Note: Python3.6+CUDA+CUDNN+Pycharm+Pytorch
# @Version: 1.0
import torch.nn as torchnn
import Gram_Matrix
class Calculation_of_Style_Loss(torchnn.Module):
    def __init__(self, value_of_target, value_of_weight):
        super(Calculation_of_Style_Loss, self).__init__()
        #Weight is similar to the content function and represents the weight beta
        self.value_of_weight = value_of_weight
        # The targer represents the layer target. The style that the new image wants to have
        # That is to save the target style
        self.value_of_target = value_of_target.detach() * self.value_of_weight
        self.gram_matrix = Gram_Matrix.Gram_Matrix_for_style_features()
        self.criterion = torchnn.MSELoss()
    def forward(self, input_value):
        #The Gram matrix of weighted input
        G_Image = self.gram_matrix(input_value) * self.value_of_weight
        #Calculate the style loss between the real style and the desired style
        self.loss = self.criterion(G_Image, self.value_of_target)
        result = input_value.clone()
        return result
    #Backward propagation
    def backward(self, retain_variabels=True):
        self.loss.backward(retain_graph=retain_variabels)# use retain_graph
        result_of_loss_value_from_backward=self.loss#return the backward value
        return result_of_loss_value_from_backward# return the result of loss
    def crossover(self,input_value,retain_variabels=True):
        G_Image = self.gram_matrix(input_value) * self.value_of_weight # The Gram matrix of weighted input
        self.loss = self.criterion(G_Image, self.value_of_target)#Calculate the style loss between the real style and the desired style
        self.loss=self.loss.backward(retain_graph=retain_variabels)
        result=self.loss#return the result
        return result



