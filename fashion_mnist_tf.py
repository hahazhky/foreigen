# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("./data", one_hot=True)  # 下载并加载mnist数据
x = tf.placeholder(tf.float32, shape=[None, 784])  # 输入的数据占位符
y_actual = tf.placeholder(tf.float32, shape=[None, 10])  # 输入的标签占位符


# 定义一个函数，用于初始化所有的权值 W
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 定义一个函数，用于初始化所有的偏置项 b
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义一个函数，用于构建卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 定义一个函数，用于构建池化层
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 构建网络
x_image = tf.reshape(x, [-1, 28, 28, 1])  # 转换输入数据shape,以便于用于网络中
W_conv1 = weight_variable([3, 3, 1, 64])
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 第一个卷积层

W_conv2 = weight_variable([3, 3, 64, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)  # 第二个卷积层
h_pool1 = max_pool(h_conv2)  # 第一个池化层

W_conv3 = weight_variable([3, 3, 64, 128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)  # 第三个卷积层

W_conv4 = weight_variable([3, 3, 128, 128])
b_conv4 = bias_variable([128])
h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)  # 第四个卷积层
h_pool2 = max_pool(h_conv4)  # 第二个池化层

W_conv5 = weight_variable([3, 3, 128, 256])
b_conv5 = bias_variable([256])
h_conv5 = tf.nn.relu(conv2d(h_pool2, W_conv5) + b_conv5)  # 第五个卷积层

W_conv6 = weight_variable([3, 3, 256, 256])
b_conv6 = bias_variable([256])
h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)  # 第六个卷积层


avg_pool = tf.nn.avg_pool(h_conv6, ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding='SAME')  # 全局池化
h_flat = tf.reshape(avg_pool, [-1, 256])  # reshape成向量

W_fc1 = weight_variable([256, 10])
b_fc1 = bias_variable([10])
y_predict = tf.nn.softmax(tf.matmul(h_flat, W_fc1) + b_fc1)  # softmax层

'''
h_flat = tf.reshape(h_conv6, [-1, 256 * 7 * 7])
W_fc1 = weight_variable([256 * 7 * 7, 2048])
b_fc1 = bias_variable([2048])
h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)  # 第一个全连接层

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # dropout层

W_fc2 = weight_variable([2048, 10])
b_fc2 = bias_variable([10])
y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # softmax层
'''
cross_entropy = -tf.reduce_mean(y_actual*tf.log(y_predict))
# cross_entropy = tf.reduce_mean(
#      tf.nn.softmax_cross_entropy_with_logits(labels=y_actual, logits=y_predict))
train_step = tf.train.AdamOptimizer(5e-4).minimize(cross_entropy)  # Adam
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

best_acc = 0
for i in range(1, 20001):
    train_batch = mnist.train.next_batch(200)
    if i % 500 == 0:  # 训练100次，验证一次
        train_acc = accuracy.eval(feed_dict={x: train_batch[0], y_actual: train_batch[1],
                                             # keep_prob: 1.0
                                             })
        print('step', i, 'training accuracy', train_acc)

        pos_num = 0
        for j in range(10):
            test_acc = accuracy.eval(feed_dict={x: mnist.test.images[1000 * j: 1000 * (j + 1)],
                                                y_actual: mnist.test.labels[1000 * j: 1000 * (j + 1)],
                                                # keep_prob: 1.0
                                                })
            pos_num += test_acc * 1000

        test_acc = pos_num / 10000
        print("test accuracy: %.4f" % test_acc)
        if best_acc < test_acc:
            best_acc = test_acc

    train_step.run(feed_dict={x: train_batch[0], y_actual: train_batch[1],
                              # keep_prob: 0.5
                              })

print('% .4f' % best_acc)


