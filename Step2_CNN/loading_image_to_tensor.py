# @author: Xuchen Sun
# @license: General Public License （GPL）
# @contact: xuchens@mun.ca
# @Hardware Note: CPU:AMD3900
# @Hardware Note: GPU: EVGA GTX1080ti
# @Software Note: Python3.6+CUDA+CUDNN+Pycharm+Pytorch
# @Version: 1.0

import PIL.Image as Image_from_PIL
import torchvision.transforms as transforms_from_torchvision


#Cons define
SIZE_IMAGE = 512
#Set EPOCH times
EPOCH=10
#define loading function: load iamge to tensor in Pytorch
def load_img_to_tensor(direct_img_path):
    image_after_open = Image_from_PIL.open(direct_img_path).convert('RGB')# use PIL.image to convert RGB pictures
    image_after_resize = image_after_open.resize((SIZE_IMAGE, SIZE_IMAGE))# change the size of picture into 512*512
    image_to_tensor = transforms_from_torchvision.ToTensor()(image_after_resize)#load image to tensor

    # because the input of CNN is 4
    result = image_to_tensor.unsqueeze(0)# use unsqueeze
    print("Loading Operation Successfully")
    return result# return the result of images in tensor type