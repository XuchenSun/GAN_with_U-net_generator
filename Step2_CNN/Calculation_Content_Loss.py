# @author: Xuchen Sun
# @license: General Public License （GPL）
# @contact: xuchens@mun.ca
# @Hardware Note: CPU:AMD3900
# @Hardware Note: GPU: EVGA GTX1080ti
# @Software Note: Python3.6+CUDA+CUDNN+Pycharm+Pytorch
# @Version: 1.0
import torch.nn as torchnn# import the module class in torch.nn

# define the calculation for content loss
class Calculation_of_Content_Loss(torchnn.Module):# torchnn module is necessary.
    #target means Content ，input means G，weight means alpha
    def __init__(self, value_of_target, value_of_weight):# initialize the value
        super(Calculation_of_Content_Loss, self).__init__()# use python keywords super
        self.value_of_weight = value_of_weight # transfer the value of weight
        # detach means target. It can calculate the gradient dynamically
        # target Represents the target content, that is, the content you want to become
        self.value_of_target = value_of_target.detach() * self.value_of_weight# use python keywords detach to get the value of target
        self.criterion = torchnn.MSELoss()# use MSELoss to get the criterion
    def forward(self, input):# define moving forward calculation
        self.loss = self.criterion(input * self.value_of_weight, self.value_of_target)#calculate the loss value
        result_of_input_clone = input.clone()# clone the value of input
        return result_of_input_clone #return the value of input
    def forwarda_half(self, input):# define moving forward calculation
        self.loss = self.criterion(input * self.value_of_weight, self.value_of_target)#calculate the loss value
        result_of_input_clone = input.clone()# clone the value of input
        result_of_input_clone =result_of_input_clone/2# half
        return result_of_input_clone #return the value of input
    def backward(self, retain_variabels=True):#define the moving backward calculation
        self.loss.backward(retain_graph=retain_variabels)# retain graph is necessary
        result_of_loss_value=self.loss
        return result_of_loss_value
    def backward_half(self, retain_variabels=True):#define the moving backward calculation
        self.loss.backward(retain_graph=retain_variabels)# retain graph is necessary
        result_of_loss_value=self.loss/2 #half
        return result_of_loss_value
    def interior_product(self, input,retain_variabels=True):
        self.loss = self.criterion(input * self.value_of_weight, self.value_of_target)#calculate the loss value
        self.loss.backward(retain_graph=retain_variabels)# retain graph is necessary
        result_of_loss_value = self.loss#calculate the loss value
        result_of_input_clone = input.clone
        self.loss=result_of_input_clone*result_of_loss_value# interior_product



