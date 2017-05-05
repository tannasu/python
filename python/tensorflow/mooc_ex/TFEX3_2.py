import tensorflow as tf
import os
import tf_test.TFUtils as tfu
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_data(path=None):
    if path == None:
        return None
    mat_files = tfu.load_mat(path)
    y_data = mat_files["y"]
    X_data = mat_files["X"]
    return X_data, y_data

def load_weight(path = None):
    if path == None:
        return None
    mat_files = tfu.load_mat(path)
    theata1 = mat_files['Theta1']
    theata2 = mat_files['Theta2']
    return theata1, theata2


def model(x, W, b):
    return tf.add(tf.matmul(x, W), b)

def single_layer_option(x, theata, logic=None):
    tf_ones = tf.ones((x.shape[0],1))
    x_ones = tf.cast(tf.concat([x, tf_ones], 1), tf.float32)
    theata_f32 = tf.cast(theata, tf.float32)
    #直接转置b
    a = tf.matmul(x_ones, theata_f32, transpose_b=True)
    z = None
    #最后一层的写外面
    if logic is None:
        z = a
    else:
        z = logic(a)
    return z

def sigmod_log_logic(x):
    return tf.log(tf.nn.sigmoid(x))


X_data, y_data = load_data("data/ex3data1.mat")
y_data = tfu.resize_y_data(y_data)
theata1, theata2 = load_weight("data/ex3weights.mat")
print(np.float32(theata1).shape)
print(np.float32(theata2).shape)


def train_loop_all(sess, x, y_, th1, th2, train_op, cost):
    cost_list = []
    for step in range(30001):
        sess.run(train_op)
        if step % 2000 == 0:
            print("step:{0}, cost:{1}".format(
                step, sess.run(cost)))
            cost_list.append(sess.run(cost))
            r_layer_2 = single_layer_option(x, th1, sigmod_log_logic)
            r_y = single_layer_option(r_layer_2, th2, sigmod_log_logic)
            correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(r_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            accuracy_val = sess.run(accuracy)
            print("accuracy_val is {0}".format(accuracy_val))

def train_loop_single(sess, x, y_, th1,th2, train_op, cost):
    keep_prob = tf.placeholder("float")
    x_shape = x.shape
    x_holder = tf.placeholder("float", [None, 400])
    y_holder = tf.placeholder("float", [None, 10])
    for i in range(x_shape[0]):
        if i % 100 == 0:
            print("step:{0}, cost:{1}".format(
                i, sess.run(cost)))
        if i%1000==0 or i == 4999:
            r_layer_2 = single_layer_option(x, th1, sigmod_log_logic)
            r_y = single_layer_option(r_layer_2, th2, sigmod_log_logic)
            correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(r_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            accuracy_val = sess.run(accuracy)
            print("accuracy_val is {0}".format(accuracy_val))
            #train_accuracy = sess.run(accuracy, feed_dict={x_holder: x[i], y_holder: y_[i], keep_prob: 1.0})
            #print(train_accuracy)
        sess.run(train_op, feed_dict={x_holder: x[i:i+1, :], y_holder: y_[i:i+1,:], keep_prob: 1.0})



def train_option(X_data, y_data, theata1, theata2):
    x = np.float32(X_data)
    y_ = np.float32(y_data)
    #设置变量
    th1 = tf.Variable(np.float32(theata1))
    th2 = tf.Variable(np.float32(theata2))
    #th1 = tf.Variable(tf.zeros([25, 401]))
    #th2 = tf.Variable(tf.zeros([10, 26]))
    print("shapes: x-{0},y-{1},th1{2}, th2{3}".format(x.shape, y_.shape, th1.shape, th2.shape))
    layer_2 = single_layer_option(x, th1, sigmod_log_logic)
    layer_3 = single_layer_option(layer_2, th2)
    cost = tf.reduce_mean((tf.nn.sigmoid_cross_entropy_with_logits(logits=layer_3, labels=y_, name="ex3_logic")))
    print(cost)
    train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    train_loop_all(sess, x, y_, th1, th2, train_op, cost)
    #train_loop_single(sess, x, y_, th1, th2, train_op, cost)
    #print(y_[0:1, :].shape)






train_option(X_data, y_data, theata1, theata2)
