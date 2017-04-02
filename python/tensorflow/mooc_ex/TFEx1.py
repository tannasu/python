import tensorflow as tf
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

with open('data/ex1data1.txt') as csv_file:
    ex1_data = list(csv.reader(csv_file))

length = len(ex1_data)
#print(type(ex1_data))
x_list = []

for i in range(length):
    x_list.append(ex1_data[0])

def create_data(csv_data, is_X=True):
    data_list = []
    sub_length = len(csv_data[0])
    if is_X:
        from_a = 0
        from_b = sub_length - 1
    else:
        from_a = sub_length - 1
        from_b = sub_length
    for sub_data in csv_data:
        data_list.append(sub_data[from_a:from_b])
    return data_list

def model(x, W, b):
    return tf.matmul(x, W) + b

x_data = np.float32(create_data(ex1_data))
#x_data_1 =  np.ones([1, length], np.float32)
#x_data = np.concatenate(([x_data], [x_data_1]),axis=0).T
##print(x_data)
y_data = np.float32(create_data(ex1_data, False))

len(x_data)
#print(np.arange(0,10,0.1) )
#graph
fig = plt.gcf()
ax = fig.add_subplot(1, 1, 1)
t = ax.set_title("TF-EX1")


def plt_graph_point(ax, x_data, y_data):
    l = ax.scatter([x_data], [y_data])



def plt_graph_line(ax, W, b):
    x = []
    y = []
    for i in range(0, 20):
        x.append(i)
        y.append(i * W + b)
    ax.plot(x,y)

def tf_options(x_data, y_data):
    #x = tf.placeholder(tf.float32,[length, 2])
    x = x_data
    #W = tf.Variable(tf.zeros([2,1]))
    W = tf.Variable(tf.zeros([1,1]))
    b = tf.Variable(tf.zeros([1]))
    #p=X*theta;
    y = model(x, W, b)

    #y_ = tf.placeholder("float", [None, 1])
    y_ = y_data
    #cost=(p-y).^2;
    cost = tf.reduce_mean(tf.square(y - y_))

    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for step in range(1500):
        #sess.run(train_op, feed_dict={x:x_data, y_:y_data})
        sess.run(train_op)
        #if step % 300 == 0:
            #print(step, sess.run(W), sess.run(b))
            #plt_graph_line(ax, sess.run(W)[0][0], sess.run(b)[0])
    return sess.run(W),sess.run(b)

plt_graph_point(ax, x_data, y_data)
Wre, bre = tf_options(x_data, y_data)
plt_graph_line(ax, Wre[0][0], bre[0])
print("w is: ",Wre, "y is :",bre)


plt.show()

