import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

BATCH_SIZE = 50
learning_rate = 0.001       

def get_real_test():
    file_dir = '/home/emin/Temp/Tensorflow/num1'
    image_list = []
    label_list = []
    for file in os.listdir(file_dir):
        image_list.append(file_dir + '/' +file)
        label_list.append(int(file[0]))
    for i in range(0,10):
        image1 = Image.open(image_list[i])
        image1 = image1.resize([28, 28])
        plt.imshow(image1)
        images1 = np.array(image1)[:,:,0]
        images1=images1.astype(float)
        img = images1.reshape(1,784)
        if i==0:
            img_last=img
        else:
            img_last=np.vstack((img_last,img))
    print(label_list, 'real number')
    return img_last

def train():
    mnist = input_data.read_data_sets('/home/emin/Temp/Tensorflow/Me/lenet5/data', one_hot=True)  # they has been normalized to range (0,1)

    test_x = mnist.test.images[:2000]
    test_y = mnist.test.labels[:2000]
    tf_x = tf.placeholder(tf.float32, [None, 28*28]) / 255.
    image = tf.reshape(tf_x, [-1, 28, 28, 1])           
    tf_y = tf.placeholder(tf.int32, [None, 10])           
    
    conv1 = tf.layers.conv2d(image, 32, 5, 1, 'same', activation=tf.nn.relu) 
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2) 
    conv2 = tf.layers.conv2d(pool1, 64, 5, 1, 'same', activation=tf.nn.relu)  
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)   
    flat = tf.reshape(pool2, [-1, 7*7*64])        
    output = tf.layers.dense(flat, 10)          
    
    loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
        labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]
    
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
    sess.run(init_op)     # initialize var in graph
    saver = tf.train.Saver() 
    for step in range(1000):
        b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
        _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
        if step % 50 == 0:
            accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y})
            print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)
    saver.save(sess, "my_net/save_net.ckpt")
            
def test():
    tf_x = tf.placeholder(tf.float32, [None, 28*28]) / 255.
    image = tf.reshape(tf_x, [-1, 28, 28, 1])           
    conv1 = tf.layers.conv2d(image, 32, 5, 1, 'same', activation=tf.nn.relu) 
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2) 
    conv2 = tf.layers.conv2d(pool1, 64, 5, 1, 'same', activation=tf.nn.relu)  
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)   
    flat = tf.reshape(pool2, [-1, 7*7*64])        
    output = tf.layers.dense(flat, 10)           
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, "my_net/save_net.ckpt")
    test_output = sess.run(output, {tf_x: get_real_test()})
    pred_y = np.argmax(test_output, 1)
    print(pred_y, 'prediction number')   
    
def main():                         # first train then restart your program and test
    train()
#    test()                  

if __name__ == '__main__':
    main()