import tensorflow as tf
import os
import tf_test.TFUtils as tfu
import matplotlib.pyplot as plt
import numpy as np
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load(path=None):
    if path == None:
        return None
    y_data, X_data = tfu.load_mat(path)
    return X_data, y_data


def resize_y_data(y_data):
    y_len = len(y_data)
    y_data_b = np.zeros([y_len, 10])
    for _i in range(y_len):
        _val = y_data[_i]
        if _val < 10:
            y_data_b[_i][_val] = 1
        else:
            y_data_b[_i][0] = 1
    return y_data_b





def display_num(data_list, single_weight, img_block_size, ax):
    img_data = tfu.stack_img_data(data_list, single_weight, img_block_size)
    ax.imshow(img_data, cmap='gray')
    return

def model(x, W, b):
    return tf.add(tf.matmul(x, W), b)


def tf_options(x_data, y_data):
    x = np.float32(x_data)
    y_ = np.float32(y_data)
    size = x.shape[1]
    y_size = y_.shape[1]
    print("x size is:{0}, y size is {1}".format(size, y_size))
    W = tf.Variable(tf.zeros([size,y_size], name="W_zero"))
    b = tf.Variable(tf.zeros([y_size]))
    p_logic = model(x, W, b)

    #J=sum(log(p).*(-y)-log(1-p).*(1-y))./m;
    #这里是求loss所以是reduce_mean，如果是求交叉熵，则用reduce_sum
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_logic, labels=y_, name="ex2_logic"))
    #cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_logic, labels=y_, name="ex2_logic"))

    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    cost_list = []

    result_w = []
    result_b = []
    for step in range(20000):

        sess.run(train_op)

        if step % 2000 == 0:
            print("step:{0}, cost:{1}".format(
                step, sess.run(cost)))
            cost_list.append(sess.run(cost))
            r_logic = model(x, W, b)
            r_y = tf.log(tf.sigmoid(r_logic))
            correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(r_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            accuracy_val = sess.run(accuracy)
            print("accuracy_val is {0}".format(accuracy_val))
        result_w = sess.run(W)
        result_b = sess.run(b)

    return result_w,result_b,cost_list, accuracy_val





fig = plt.gcf()
ax = fig.add_subplot(1,1,1)
X_data, y_data = load("data/ex3data1.mat")
y_data_r = resize_y_data(y_data)

#随机100个数字灰度数据
random_num = [X_data[x] for x in [random.randint(0, len(y_data)-1) for r in range(0,100)]]
#显示灰度图
display_num(random_num,20, 10, ax)
#逻辑回归计算
re_w, re_b, cost_list,accuracy_val = tf_options(X_data, y_data_r)

plt.show()
