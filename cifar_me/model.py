import tensorflow as tf

def inference(images, batch_size, n_classes,keep_prob):  
    conv1 = tf.layers.conv2d(images,6,3,1,'valid',activation=tf.nn.relu)  
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2)
    conv2 = tf.layers.conv2d(pool1, 16, 3, 1, 'valid', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
    reshape = tf.reshape(pool2, shape=[batch_size, -1])
    local3 = tf.layers.dense(reshape, 400)
    local4 = tf.layers.dense(local3, 400) 
    h_drop = tf.nn.dropout(local4, keep_prob)
    softmax_linear = tf.layers.dense(h_drop, n_classes)
    return softmax_linear

#%%
def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss

#%%
def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op

#%%
def evaluation(logits, labels):
  with tf.variable_scope('accuracy') as scope:
      correct = tf.nn.in_top_k(logits, labels, 1)
      correct = tf.cast(correct, tf.float16)
      accuracy = tf.reduce_mean(correct)
      tf.summary.scalar(scope.name+'/accuracy', accuracy)
  return accuracy

#%%