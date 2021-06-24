# @author: Xuchen Sun
# @license: General Public License （GPL）
# @contact: xuchens@mun.ca
# @Hardware Note CPU:AMD3900
# @Hardware Note GPU: EVGA GTX1080ti
# @Software Note: Python3.6+CUDA+CUDNN+Pycharm+Tensorflow
# @Version: 1.2
# @Date: 2021-05-10
# @Note: This function is only used for rename the images from three used folders, and these images' names must be int and start from 1.
import os
#path='test'
#path='train'
path = 'val' # if using images from val, this command should be used

# get the all file and save to list
fileList = os.listdir(path)

n = 0
for i in fileList:
    # set the name of old file（path+file name）
    oldName = path + os.sep + fileList[n]  # set up new name

    # set new file begin with 1
    newName = path + os.sep + str(n + 1) + '.JPG'# set up new names

    os.rename(oldName, newName)  # use renmae function to change the name in file
    print(oldName, '======>', newName)# print new names

    n += 1# set n increaing by iteration