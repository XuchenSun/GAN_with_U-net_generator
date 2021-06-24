# @author: Xuchen Sun
# @license: General Public License （GPL）
# @contact: xuchens@mun.ca
# @Hardware Note CPU:AMD3900
# @Hardware Note GPU: EVGA GTX1080ti
# @Software Note: Python3.6+CUDA+CUDNN+Pycharm+Tensorflow
# @Version: 1.2666
# @Date: 2021-05-10

import define_GAN


gan1=define_GAN.GAN("gan1",150)
gan1.print_summary_of_G_and_D()
gan1.start_train()
gan2=define_GAN.GAN("gan2",200)
gan2.start_train()