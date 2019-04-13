import os
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt

file_dir = '/home/emin/Temp/Tensorflow/data/cifar-10/train'
classes_list=[]
def load_img(path):
    img = skimage.io.imread(path)
    img = img / 255.0
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))[None, :, :, :]   # shape [1, 224, 224, 3]
    return resized_img

def get_files():
    image_list = []
    label_list= []
    num = 0
    for file in os.listdir(file_dir):
        filedir_f = file_dir +'/' +file
        for filedir_p in os.listdir(filedir_f):
            image_list.append(filedir_f + '/' +filedir_p)
            label_list.append(num)
        num = num + 1
        classes_list.append(file)
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)   
    
    all_image_list = temp[:, 0]
    all_label_list = temp[:, 1]
    all_label_list = [int(float(i)) for i in all_label_list]
    print('There are %d image\n' %(len(image_list)))
    print('There are %d classes\n' %(num))
    all_image = []
    for i in range(len(all_image_list)):
        try:
            resized_img = load_img(all_image_list[i])
        except OSError:
            continue
        all_image.append(resized_img) 
        if len(all_image) == 300:        
              all_label_list = all_label_list[0:300]  
              break
    all_image = (np.array(all_image)).reshape(-1,224,224,3)
    all_label_list = (np.array(all_label_list))#.reshape(-1,1)
    return all_image,all_label_list

class Vgg16:
    vgg_mean = [103.939, 116.779, 123.68]

    def __init__(self, vgg16_npy_path=None, restore_from=None):
        # pre-trained parameters
        try:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        except FileNotFoundError:
            print('Please download VGG16 parameters from here https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM\nOr from my Baidu Cloud: https://pan.baidu.com/s/1Spps1Wy0bvrQHH2IMkRfpg')

        self.tfx = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.tfy = tf.placeholder(tf.int64, [None])


        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.tfx * 255.0)
        bgr = tf.concat(axis=3, values=[
            blue - self.vgg_mean[0],
            green - self.vgg_mean[1],
            red - self.vgg_mean[2],
        ])

        # pre-trained VGG layers are fixed in fine-tune
        conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3, 'pool3')

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4 = self.max_pool(conv4_3, 'pool4')

        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        pool5 = self.max_pool(conv5_3, 'pool5')

        # detach original VGG fc layers and
        # reconstruct your own fc layers serve for your own purpose
        self.flatten = tf.reshape(pool5, [-1, 7*7*512])
        self.fc6 = tf.layers.dense(self.flatten, 256, tf.nn.relu, name='fc6')
        self.out = tf.layers.dense(self.fc6, 10, name='out')

        self.sess = tf.Session()
        if restore_from:
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_from)
        else:   # training graph 
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=self.out, labels=self.tfy, name='xentropy_per_example')
            self.loss = tf.reduce_mean(self.cross_entropy, name='loss')
            self.correct = tf.nn.in_top_k(self.out, self.tfy, 1)
            self.correct = tf.cast(self.correct, tf.float16)
            self.accuracy = tf.reduce_mean(self.correct)
            self.train_op = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)
            self.sess.run(tf.global_variables_initializer())

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
            conv = tf.nn.conv2d(bottom, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1]))
            return lout

    def train(self, x, y):
        loss, _ = self.sess.run([self.loss, self.train_op], {self.tfx: x, self.tfy: y})
        return loss

    def predict(self, paths):
        fig, axs = plt.subplots(1, 2)
        for i, path in enumerate(paths):
            x = load_img(path)
            classes = (np.argmax((self.sess.run(self.out, {self.tfx: x})) , axis=1))[0]
            axs[i].imshow(x[0])
            axs[i].set_title(classes_list[classes])
            axs[i].set_xticks(()); axs[i].set_yticks(())
        plt.show()

    def save(self, path='/home/emin/Temp/Tensorflow/Me/vgg VS transfer/transfer_learn'):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)


def train():
    image_x, label_y = get_files()
    vgg = Vgg16(vgg16_npy_path='/home/emin/Temp/Tensorflow/Peking University Class/vgg/vgg16.npy')
    print('Net built')
    for i in range(50):
        b_idx = np.random.randint(0, len(image_x), 20)
        train_loss = vgg.train(image_x[b_idx], label_y[b_idx])
        print(i, 'train loss: ', train_loss)
    vgg.save('/home/emin/Temp/Tensorflow/Me/vgg VS transfer/transfer_learn')      # save learned fc layers

def eval():
    _,_=get_files()
    vgg = Vgg16(vgg16_npy_path='/home/emin/Temp/Tensorflow/Peking University Class/vgg/vgg16.npy',
                restore_from='/home/emin/Temp/Tensorflow/Me/vgg VS transfer/transfer_learn')
    vgg.predict(
        ['/home/emin/Temp/Tensorflow/data/cifar-10/test/cat/batch_1_num_896.jpg','/home/emin/Temp/Tensorflow/data/cifar-10/test/horse/batch_1_num_13.jpg'])


if __name__ == '__main__':
    # download()
    train()
    #eval()