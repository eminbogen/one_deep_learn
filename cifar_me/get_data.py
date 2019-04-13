import tensorflow as tf
import numpy as np
import os
import math

def get_files(file_dir):
    image_list = []
    label_list= []
    num = 0
    for file in os.listdir(file_dir):
        filedir_f = file_dir + file
        for filedir_p in os.listdir(filedir_f):
            image_list.append(filedir_f + '/' +filedir_p)
            label_list.append(num)
        num = num + 1
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)   
    
    all_image_list = temp[:, 0]
    all_label_list = temp[:, 1]
    all_label_list = [int(float(i)) for i in all_label_list]
    print('There are %d image\n' %(len(image_list)))
    return all_image_list,all_label_list

def get_batch(image, label, image_W, image_H, batch_size, capacity):   
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)    
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch