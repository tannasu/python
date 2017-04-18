import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import tf_test.TFUtils as tfu
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ex1_data = tfu.open_csv("data/ex2data1.txt")
ex2_data = tfu.open_csv("data/ex2data2.txt")
fig = plt.gcf()

def format_x_data_small(x_data):
    return x_data * 0.01

def create_cycle_x_data(x_data):
    x_data_2 = np.square(x_data)
    return np.column_stack((x_data, x_data_2))

def plt_graph_point(ax, x_data, y_data):
    zero_list = [[],[]]
    one_list = [[],[]]
    for i in range(len(x_data)):
        if y_data[i][0] == 0:
            use_list = zero_list
        else:
            use_list = one_list
        use_list[0].append(x_data[i][0])
        use_list[1].append(x_data[i][1])
    ax.scatter([zero_list[0]], [zero_list[1]], marker='o')
    ax.scatter([one_list[0]], [one_list[1]], marker='+')


def plt_graph_line(ax, W1, W2, b):
    x = []
    y = []
    print(W1, W2)
    for i in range(0, 20):
        x.append(i)
        y.append(i * W1/W2 + b)
    ax.plot(x,y)

def model(x, W, b=0):
    return tf.matmul(x, W)

def model2(x, W, b):
    return tf.add(tf.matmul(x, W), b)

NEAR_0 = 1e-5

def nan_to_num(n):
    return tf.clip_by_value(n, NEAR_0, 1)



def tf_options(x_data, y_data):
    #x = tf.placeholder(tf.float32,[length, 2])
    x = x_data
    size = x.shape[1]
    W = tf.Variable(tf.zeros([size,1], name="W_zero"))
    W = tf.Variable(tf.random_normal([size, 1], stddev=0.01, name="W_random"))
    b = tf.Variable(tf.zeros([1]))
    p_logic = model(x, W)
    y_ = y_data
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
    for step in range(3000):

        sess.run(train_op)

        if step % 500 == 0:
            print("step:{0}, W:{1}, b:{2}, cost:{3}".format(
                step, sess.run(W), sess.run(b), sess.run(cost)))
            cost_list.append(sess.run(cost))
        if step > 2 and last_cost < sess.run(cost):
            pass
        result_w = sess.run(W)
        result_b = sess.run(b)
        last_cost = sess.run(cost)
    print(sess.run(W))
    return result_w,result_b,cost_list, result_b


x_data = np.float32(tfu.create_data(ex1_data))
x_data = format_x_data_small(x_data)
y_data = np.float32(tfu.create_data(ex1_data, is_X=False))
ax1 = fig.add_subplot(1, 2, 1)
plt_graph_point(ax1, x_data, y_data)
Wre, bre, cost_list, w_list = tf_options(x_data, y_data)
print(cost_list)
#plt_graph_line(ax1, Wre[0][0], Wre[1][0],bre)

ax2 = fig.add_subplot(1, 2,2)
x_data = np.float32(tfu.create_data(ex2_data))
y_data = np.float32(tfu.create_data(ex2_data, is_X=False))
x_data_square = create_cycle_x_data(x_data)
plt_graph_point(ax2, x_data, y_data)
Wre, bre, cost_list, w_list = tf_options(x_data_square, y_data)
print(Wre, bre)
print(cost_list)
plt.show()


