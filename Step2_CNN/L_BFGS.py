# @author: Xuchen Sun
# @license: General Public License （GPL）
# @contact: xuchens@mun.ca
# @Hardware Note: CPU:AMD3900
# @Hardware Note: GPU: EVGA GTX1080ti
# @Software Note: Python3.6+CUDA+CUDNN+Pycharm+Pytorch
# @Version: 1.0
import torch.nn as torchnn
import torch.optim as optim_from_pytorch
#define the optimier with L_BFGS algorithms
result_of_output=None
def optimier_with_L_BFGS_Algorithms(input_img):
    #Input_ The value of img is transformed into the parameter type of neural network
    result_of_input_param = torchnn.Parameter(input_img.data)
    # Tell the optimizer that we optimize the input_ IMG, not the weight of network layer
    # Using lbfgs optimizer
    result_of_output=torchnn.MSELoss
    result_of_optimizer = optim_from_pytorch.LBFGS([result_of_input_param])
    return result_of_input_param, result_of_optimizer
