import os
from PIL import Image
import PIL.ImageOps
file_dir = '/home/emin/Temp/Tensorflow/num' 
image_list = []
label_list = []
for file in os.listdir(file_dir):
    image_list.append(file_dir + '/' +file)
    label_list.append((file[0]))
for i in range(0,10):
    image1 = Image.open(image_list[i])
    image1.save('/home/emin/Temp/Tensorflow/num1/' + label_list[i]+ '.jpg')    
file_dir1 = '/home/emin/Temp/Tensorflow/num1' 
image_list1 = []
label_list1 = []
for file in os.listdir(file_dir1):
    image_list1.append(file_dir1 + '/' +file)
    label_list1.append((file[0]))    
for i in range(0,10):
    image1 = Image.open(image_list1[i])
    inverted_image = PIL.ImageOps.invert(image1)
    inverted_image.save('/home/emin/Temp/Tensorflow/num1/' + label_list1[i]+ '.jpg')