# @author: Xuchen Sun
# @license: General Public License （GPL）
# @contact: xuchens@mun.ca
# @Hardware Note CPU:AMD3900
# @Hardware Note GPU: EVGA GTX1080ti
# @Software Note: Python3.6+CUDA+CUDNN+Pycharm+Tensorflow
# @Version: 1.2
# @Date: 2021-05-10

# @Note: This python file is used for rotating image into correct positions. Mask pictures should be in the left and targets' images should be in right
# pip install glob2

import cv2
from glob2 import glob

for file in glob('*.jpg'): #select all image from the folder
    original_image=cv2.imread(file)# read image
    horizontal_image=cv2.flip(original_image,1)# flip image
    split_Name=file.split(".")# set the name
    newName=split_Name[0]# set new name
    cv2.imwrite(newName+'.jpg',horizontal_image)# write images with new name