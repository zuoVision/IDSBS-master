
import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn import preprocessing

# 添加层
def add_layer(inputs, Weights, biases, activation_function=None):
    # add one more layer and return the output of this layer
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def bp_train(individual,net,dataPath,lr,epoch):
    df =  pd.read_csv(dataPath)
    df =  df.values
    df = np.array(df)
    x = df[:,0:net[0]]
    y = df[:,net[0]:]
    # print('x:\t',x.shape)
    # print('y:\t',y.shape)

    #正常归一化及还原，精度0.01
    scaler_x = preprocessing.MinMaxScaler()
    x_data = scaler_x.fit_transform(x)
    scaler_y = preprocessing.MinMaxScaler()
    y_data = scaler_y.fit_transform(y)

    xs = tf.placeholder(tf.float32, [None, net[0]])
    ys = tf.placeholder(tf.float32, [None, net[2]])

    w1_len = net[0]*net[1]
    w2_len = net[1]*net[2]
    b1_len = net[1]
    b2_len = net[2]
    w1 = []

    for i in range(net[0]):
        a = individual[i*net[1]:i*net[1]+net[1]]
        w1.append(a)
        # print('a:',a)
        # print('w1:',w1)
    weight = individual[net[0]*net[1]:net[0]*net[1]+net[1]*net[2]]
    w2 = []
    for i in range(net[1]):
        a = weight[i*net[2]:i*net[2]+net[2]]
        w2.append(a)
        # print('a:', a)
        # print('w2:', w2)

    b1 = individual[w1_len+w2_len:w1_len+w2_len+b1_len]
    b2 = individual[-b2_len:]

    Weights_1 = tf.Variable(w1, dtype=tf.float32,name='Weights_1')
    biases_1 = tf.Variable(b1, dtype=tf.float32,name='biases_1')
    Weights_2 = tf.Variable(w2, dtype=tf.float32,name='Weights_2')
    biases_2 = tf.Variable(b2, dtype=tf.float32,name='biases_2')

    # 3.定义神经层：隐藏层和预测层
    # add hidden layer 输入值是 xs，在隐藏层有 10 个神经元
    l1 = add_layer(xs, Weights_1, biases_1, activation_function=tf.nn.sigmoid)
    # add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
    prediction = add_layer(l1, Weights_2, biases_2, activation_function=None)

    # 4.定义 loss 表达式
    # the error between prediciton and real data
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))
    # 5.选择 optimizer 使 loss 达到最小
    # 这一行定义了用什么方式去减少 loss，学习率是 0.1
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    # 上面定义的都没有运算，直到 sess.run 才会开始运算
    sess.run(init)
    # 迭代 1000 次学习，sess.run optimizer
    for i in range(epoch):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        error = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
    # print('适应度：',1/error)
    return 1/error

if __name__ == '__main__':
    pass
            