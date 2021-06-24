# @author: Xuchen Sun
# @license: General Public License （GPL）
# @contact: xuchens@mun.ca
# @Hardware Note: CPU:AMD3900
# @Hardware Note: GPU: EVGA GTX1080ti
# @Software Note: Python3.6+CUDA+CUDNN+Pycharm+Pytorch
# @Version: 1.0
import torch #use torch package
import torch.nn as torchnn
#define the Gram Matrix to calculate the style features
class Gram_Matrix_for_style_features(torchnn.Module):
    def __init__(self):
        super(Gram_Matrix_for_style_features, self).__init__()
    def forward(self, input):
        x1, x2, x3, x4 = input.size()#use 4 varaible
        # change the feature map to 2D
        features = input.view(x1 * x2, x3 * x4)# get the features
        # The calculation method of inner product is to multiply the characteristic graph by its inverse
        gram_matrix = torch.mm(features, features.t())
        # get the average of result
        gram_matrix /= (x1 * x2 * x3 * x4)#get the gram matrix which includes the average result
        return gram_matrix#return the result
    def backward(self, input_image):
        # normalize img
        return (input_image - self.mean) / self.std# normalize the image