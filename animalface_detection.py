'''
Reference : https://ieeexplore.ieee.org/document/4147155
Implement : tensorflow 2.0, numpy, keras 2.0+,

Created by JunWoo Kim,
Date : 2019.4.3
'''
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageOps
from PIL import Image
from glob import glob


'''Default path'''
train_path = '/home/cheeze/PycharmProjects/KJW/capstone_project/Image/'
image_files = os.listdir(train_path)
train_path_name = os.path.join(train_path, image_files[1])
total_file_paths = []

'''Get a whole paths : total_file_paths'''
for item in image_files:
    cur_file_path = os.path.join(train_path, item)
    total_file_paths = total_file_paths + [cur_file_path]

'''whole total_data_list'''
total_data_list = []
for item in total_file_paths:
    data_list = glob(os.path.join(train_path, item, '*'))
    total_data_list = total_data_list + data_list

'''Get Label'''
def get_label(path):
    return str(path.split('/')[-2])

label_list = get_label(total_data_list[1])



'''Get Random'''
rand_n = 100
print(total_data_list[rand_n], get_label(total_data_list[rand_n]))

'''Call Image&Label'''
path = total_data_list[rand_n]
image = np.array(Image.open(path))

def read_image(path):
   with open(path, 'rb') as f:
       with Image.open(f) as img:
           img = ImageOps.equalize(img)
           return img

'''One hot encoding through the label name'''
class_name = get_label(path)
read_image(path)




print(train_images)
#print(train_labels)
