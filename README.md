# Part0 Basic Concept

0.1 Filler

0.2 CNN

0.3 GAN



# Part1 GAN：Generator Architecture(Unet)

 ![](https://github.com/XuchenSun/GAN_with_U-net_generator/blob/main/Generator(Unet).png)
 
 
 
 
 
# Part2 GAN：Discriminator(PatchGAN)


Input
Input image and the target image, which it should classify as real.

Input image and the generated image (output of generator), which it should classify as fake.  

Concatenate these 2 inputs together in the code (tf.concat([inp, tar], axis=-1))  

Output
The output of discriminator is (batch_size, 30, 30, 1) // batch_size =1  

Every 30x30 patch in the output of PatchGAN classify a 70x70 portion of the input image(256*256)  

![](https://github.com/XuchenSun/GAN_with_U-net_generator/blob/main/Discriminator.png)



# Part3
# Part3 Training Process
URL 

https://github.com/XuchenSun/GAN_with_U-net_generator/blob/main/updates.ipynb

# Part4 Game Desgin and Image Display

4.1 Create basic UE4 objects

4.2 Create a new map, import UE4 objects and define the details of objects and characters

4.3 Set the light render to make the scene better

4.3.1 scene light desgin

4.3.2 character light desgin

4.4 creat the WebBrowser

4.5 add the dynamic objects into scene

4.6 Compile and run the application to debug.
