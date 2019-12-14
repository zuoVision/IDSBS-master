'''
Created on 2017骞�10鏈�17鏃�

@author: ljs
'''
import tensorflow as tf
import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
def add_layer(inputs, Weights, biases, activation_function=None):
    # add one more layer and return the output of this layer
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def bp_test(dataPath,net,lr,epoch):
    df = pd.read_csv(dataPath)
    df = df.values
    df = np.array(df)
    x = df[:, 0:net[0]]
    y = df[:, net[0]:]

    scaler_x = preprocessing.MinMaxScaler()
    x_data = scaler_x.fit_transform(x)
    scaler_y = preprocessing.MinMaxScaler()
    y_data = scaler_y.fit_transform(y)

    xs = tf.placeholder(tf.float32, [None, 11])
    ys = tf.placeholder(tf.float32, [None, 1])

    Weights_1 = tf.Variable(tf.random_normal([11, 10]),name='Weights_1')
    biases_1 = tf.Variable(tf.zeros([1, 10]) + 0.1,name='biases_1')
    Weights_2 = tf.Variable(tf.random_normal([10, 1]),name='Weights_2')
    biases_2 = tf.Variable(tf.zeros([1, 1]) + 0.1,name='biases_2')

    l1 = add_layer(xs, Weights_1, biases_1, activation_function=tf.nn.relu)

    prediction = add_layer(l1, Weights_2, biases_2, activation_function=None)

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, "tf_model\\bp_model.ckpt")
    for i in range(epoch):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
    #真实误差
    prediction_value = sess.run(prediction, feed_dict={xs: x_data})
    real_pre = scaler_y.inverse_transform(prediction_value)
    print(len(y))
    print(len(real_pre))
    result = y - real_pre
    re = []
    re_sum = 0
    for i in range(x.shape[0]):
        re_sum = re_sum + abs(round(float(result[i]),8))
        re.append(round(float(result[i]),8))
    print(re_sum/x.sahpe[0])