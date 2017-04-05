import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import tf_test.TFUtils as tfu

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ex1_data = tfu.open_csv('data/ex1data1.txt')

ex2_data = tfu.open_csv('data/ex1data2.txt')



def model(x, W, b):
    return tf.matmul(x, W) + b

#print(np.arange(0,10,0.1) )
#graph
fig = plt.gcf()



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
    size = x.shape[1]
    #W = tf.Variable(tf.zeros([2,1]))
    W = tf.Variable(tf.zeros([size,1]))
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
        if step % 300 == 0:
            #print(step, sess.run(cost))
            print(step, sess.run(W), sess.run(b))
            #plt_graph_line(ax, sess.run(W)[0][0], sess.run(b)[0])
    return sess.run(W),sess.run(b)

#ex1-----------------------------------------
x_data = np.float32(tfu.create_data(ex1_data))
y_data = np.float32(tfu.create_data(ex1_data, is_X=False))

ax = fig.add_subplot(1, 1, 1)
t = ax.set_title("TF-EX1-1")
plt_graph_point(ax, x_data, y_data)
Wre, bre = tf_options(x_data, y_data)
plt_graph_line(ax, Wre[0][0], bre[0])
#print("w is: ",Wre, "y is :",bre)


#ex2------------------------------------
x_data = np.float32(tfu.create_data(ex2_data))
y_data = np.float32(tfu.create_data(ex2_data, is_X=False))
#
for i in range(len(x_data)):
    x_data[i][0] = x_data[i][0] / 1000
    y_data[i] = y_data[i] / 10000
print(x_data)
print(y_data)
Wre, bre = tf_options(x_data, y_data)

print("w is: ",Wre, "y is :",bre)


plt.show()

