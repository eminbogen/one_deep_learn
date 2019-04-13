import os
import numpy as np
import tensorflow as tf
import get_data
import model
import matplotlib.pyplot as plt

N_CLASSES = 10	#类别
IMG_W = 32	#图像整合大小
IMG_H = 32
BATCH_SIZE = 25	#批次大小，一次25张
CAPACITY = 1000 #每批次最大容量
MAX_STEP = 100000 # 训练总批次
VAL_SPACE = 100	#训练几批，测试一次
learning_rate = 0.0001 # 学习率

# 用这个程序，请改路径
train_dir = '/home/emin/Temp/Tensorflow/data/cifar-10/train/' 
test_dir  = '/home/emin/Temp/Tensorflow/data/cifar-10/test/'
logs_train_dir = '/home/emin/Temp/Tensorflow/Me/cifar_me/train/'
logs_val_dir = '/home/emin/Temp/Tensorflow/Me/cifar_me/val/'
arra = [0]*(int(MAX_STEP/VAL_SPACE))#储存训练准确度
brra = [0]*(int(MAX_STEP/VAL_SPACE))#储存测试准确度
train, train_label = get_data.get_files(train_dir)	#读训练集
val, val_label     = get_data.get_files(test_dir)	#读测试集
train_batch, train_label_batch = get_data.get_batch(train,
                                              train_label,
                                              IMG_W,
                                              IMG_H,
                                              BATCH_SIZE, 
                                              CAPACITY)
val_batch, val_label_batch = get_data.get_batch(val,
                                              val_label,
                                              IMG_W,
                                              IMG_H,
                                              BATCH_SIZE, 
                                              CAPACITY)
                        							#生成批次
x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
keep_prob = tf.placeholder(tf.float32)			#损失率

logits = model.inference(x, BATCH_SIZE, N_CLASSES,keep_prob)
loss = model.losses(logits, y_)  			#计算误差
acc = model.evaluation(logits, y_)			#计算准确度
train_op = model.trainning(loss, learning_rate)		#训练开始程序
         
with tf.Session() as sess:
    saver = tf.train.Saver()				#运行队列准备等
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess= sess, coord=coord)
    summary_op = tf.summary.merge_all()        
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)
    							#储存训练信息
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():			#意外终止
                    break
            tra_images,tra_labels = sess.run([train_batch, train_label_batch])
            _, tra_loss, tra_acc = sess.run([train_op, loss, acc],
                                            feed_dict={x:tra_images, y_:tra_labels,keep_prob:0.5})
                                            		#训练进行，并获取准确度等
            if step % 20 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                					#显示每次训练时准确度
            if step % VAL_SPACE == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run([val_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, acc], 
                                             feed_dict={x:val_images, y_:val_labels,keep_prob:1.0})
                print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc*100.0))   
                					#每VAL_SPACE进行一次测试,显示准确度等
                arra[int(step/VAL_SPACE)]=tra_acc*100
                brra[int(step/VAL_SPACE)]=val_acc*100	#保存数据
            if step % 1000 == 0:    			#保存训练模型
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()           
    coord.join(threads)
    plt.plot(arra, c='blue')
    plt.show()
    plt.plot(brra, c='red')
    plt.show()						#出图