# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 17:28:37 2018

@author: Spencersun
"""
import csv
import os
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from sklearn.model_selection import train_test_split
import scipy.io as sio


np.set_printoptions(threshold=30)  
feature_o = []
with open('feature_train_480p_crf30.csv', 'r') as csv_file:
    all_lines = csv.reader(csv_file)
    for one_line in all_lines:
        feature_o.append(one_line)
feature_o = np.array(feature_o, dtype='float64')
#feature = np.array(feature)
label_o = []
with open('label_train_480p_crf30.csv', 'r') as csv_file:
    all_lines = csv.reader(csv_file)
    for one_line in all_lines:
        label_o.append(one_line)
label_o = np.array(label_o, dtype='float64')
#label = np.array(label)
#
feature_test = []
with open('feature_train_480p.csv', 'r') as csv_file:
    all_lines = csv.reader(csv_file)
    for one_line in all_lines:
        feature_test.append(one_line)
feature_test = np.array(feature_test, dtype='float64')
#feature_test = np.array(feature_test)
#
label_test = []
with open('label_train_480p.csv', 'r') as csv_file:
    all_lines = csv.reader(csv_file)
    for one_line in all_lines:
        label_test.append(one_line)
label_test = np.array(label_test, dtype='float64')
#label_test = np.array(label_test)
#
feature720_test_28 = []
with open('feature_test_480p_crf30.csv', 'r') as csv_file:
    all_lines = csv.reader(csv_file)
    for one_line in all_lines:
        feature720_test_28.append(one_line)
feature720_test_28 = np.array(feature720_test_28, dtype='float64')
#feature720_test_28= np.array(feature720_test_28)
#
label720_test_28 = []
with open('label_test_480p_crf30.csv', 'r') as csv_file:
    all_lines = csv.reader(csv_file)
    for one_line in all_lines:
        label720_test_28.append(one_line)
label720_test_28 = np.array(label720_test_28, dtype='float64')
#label720_test_28 = np.array(label720_test_28)



feature720_test = []
with open('feature_test_480p.csv', 'r') as csv_file:
    all_lines = csv.reader(csv_file)
    for one_line in all_lines:
        feature720_test.append(one_line)
feature720_test = np.array(feature720_test, dtype='float64')
#feature720_test = np.array(feature720_test)

label720_test = []
with open('label_test_480p.csv', 'r') as csv_file:
    all_lines = csv.reader(csv_file)
    for one_line in all_lines:
        label720_test.append(one_line)
label720_test = np.array(label720_test, dtype='float64')
#label720_test = np.array(label720_test)

label_init = []
ss_x = preprocessing.MinMaxScaler().fit(feature_o)
feature = ss_x.transform(feature_o)
ss_y = preprocessing.StandardScaler().fit(label_o)
label_init = ss_y.transform(label_o)  
feature720_test_init = ss_x.transform(feature720_test_28)
label720_test_init = ss_y.transform(label720_test_28) 
#
activation = tf.nn.tanh
step_size = 0.05
best_loss = 1
test_set_size = 0.2
best_acc_1 = 0
best_acc_2 = 0
best_iter = 0
save_dir = 'checkpointss'
save_path = os.path.join(save_dir, 'best_training')

def full_connection(inputs, in_size, out_size, activation_func=None, norm=True):
    Weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0.01, stddev=0.1))
    biases = tf.Variable(tf.zeros([out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    
    if activation_func is None:
        Wx_plus_b = Wx_plus_b
    else:
        Wx_plus_b = activation_func(Wx_plus_b)

    return Wx_plus_b


def batch_norm(inputs, fc_mean, fc_var, out_size,norm=True):
    if norm is True:
        scale = tf.Variable(tf.ones([out_size]),name='scale')
        shift = tf.Variable(tf.zeros([out_size]),name='shift')
        epsilon = 0.001
        Wx_plus_b = tf.nn.batch_normalization(inputs, fc_mean, fc_var, shift, scale, epsilon)
    return Wx_plus_b


train_x, val_x, train_y, val_y = train_test_split(feature, label_init, test_size=0.2)
R_val = np.reshape(val_x[:, 3], [len(val_x), 1])  # extract bitrate by log values on test set
R_train = np.reshape(train_x[:, 3], [len(train_y), 1])  # extract bitrate by log values on train set
CRF_val = np.reshape(val_x[:, -1], [len(val_x), 1])  # extract values of CRF on test set
CRF_train = np.reshape(train_x[:, -1], [len(train_y), 1])  # extract values of CRF on train set
#train_x = feature[0:int(0.8*len(feature))]
#val_x   = feature[int(0.2*len(feature))::]
#train_y = label_init[0:int(0.8*len(feature))]
#val_y = label_init[int(0.2*len(feature))::]
#R_train = np.reshape(train_x[:, 3], [len(train_x), 1])  # extract bitrate by log values on train set
#CRF_train = np.reshape(train_x[:, -1], [len(train_x), 1])  # extract values of CRF on train set
#R_val = np.reshape(val_x[:, 3], [len(val_x), 1])  # extract bitrate by log values on train set
#CRF_val = np.reshape(val_x[:, -1], [len(val_x), 1])  # extract values of CRF on train set
train_x = np.delete(train_x,-1,axis=1)
val_x = np.delete(val_x,-1,axis=1)
train_x = np.delete(train_x,0,axis=1)
val_x = np.delete(val_x,0,axis=1)
train_x = np.delete(train_x,0,axis=1)
val_x = np.delete(val_x,0,axis=1)
test_x   = feature720_test_init
test_y   = label720_test_init
test_x = np.delete(test_x,-1,axis=1)
test_x = np.delete(test_x,0,axis=1)
test_x = np.delete(test_x,0,axis=1)

# tensorflow session
xs = tf.placeholder(tf.float32, [None, 13])
ys = tf.placeholder(tf.float32, [None, 3])

Wx_plus_b_fc1 = full_connection(xs, 13, 50, activation_func = activation)
fc_mean1, fc_var1 = tf.nn.moments(Wx_plus_b_fc1, axes=[0]) 
Wx_plus_b_fc1_bn = batch_norm(Wx_plus_b_fc1, fc_mean1, fc_var1, 50,norm=True)

#Wx_plus_b_fc2 = full_connection(Wx_plus_b_fc1_bn, 100, 50, activation_func = activation)
#fc_mean2, fc_var2 = tf.nn.moments(Wx_plus_b_fc2, axes=[0]) 
#Wx_plus_b_fc2_bn = batch_norm(Wx_plus_b_fc2, fc_mean2, fc_var2, 50,norm=True)

prediction = full_connection(Wx_plus_b_fc1_bn, 50, 3, activation_func = None)

cost = tf.reduce_mean(tf.square(ys - prediction))
train_step = tf.train.GradientDescentOptimizer(step_size).minimize(cost)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for iteration in range(20+1):
    sess.run(train_step, feed_dict={xs: train_x, ys: train_y})
    loss_val = sess.run(cost, feed_dict={xs: val_x, ys: val_y})
    loss_train = sess.run(cost, feed_dict={xs: train_x, ys: train_y})
    loss_test = sess.run(cost, feed_dict={xs: test_x, ys: test_y})
    if iteration % 10 == 0:
        if loss_val < best_loss:
            best_loss = loss_val
            print('Iteration =', iteration, 'train =', loss_train, 'val =', loss_val, 'test =', loss_test, "*")
        else:
            print('Iteration =', iteration, 'train =', loss_train, 'val =', loss_val, 'test =', loss_test)
    if iteration % 10 == 0:
        prediction_abc_test  = sess.run(prediction, feed_dict={xs: test_x})
        Wx_plus_b  = sess.run(Wx_plus_b_fc1, feed_dict={xs: test_x})
        Wx_plus_b_bn  = sess.run(Wx_plus_b_fc1_bn, feed_dict={xs: test_x})
        mean1, var1  = sess.run([fc_mean1, fc_var1], feed_dict={xs: test_x})
#        mean2, var2  = sess.run([fc_mean2, fc_var2], feed_dict={xs: test_x})
#        Wx_plus_b,mea,var = sess.run([Wx_plus_b_fc2,fc_mean2, fc_var2], feed_dict={xs: test_x})
#        me,va = sess.run([mean, var], feed_dict={xs: test_x})
        
#        prediction_abc_val   = sess.run(prediction, feed_dict={xs: val_x})
#        prediction_abc_train   = sess.run(prediction, feed_dict={xs: train_x})

#        R_test = np.reshape(feature_test[:,3], [len(feature_test), 1])
#        R_test = R_test[0:int(0.7*len(feature_test))]
#        CRF_test = np.reshape(feature_test[:, -1], [len(feature_test), 1])
#        CRF_test = CRF_test[0:int(0.7*len(feature_test))]
#        prediction_tem = prediction_abc_train
#        prediction_abc = []
#        for i in range(len(prediction_tem)):
#            prediction_temt = 33*[prediction_tem[i]]
#            prediction_abc = prediction_abc + prediction_temt
#        prediction_abc = np.array(prediction_abc)   
#        
#        prediction_abc = ss_y.inverse_transform(prediction_abc)    
#        label = prediction_abc
#        
#        # generate bitrate bascis on predicted model parameters
#        R_1_t = []
#        R_2_t = []
#        bitrate_pre = []
#        for i in range(len(CRF_test)):
#            if np.square(label[i][1]) - 4*label[i][0]*(label[i][2]-CRF_test[i]) > 0:
#                R_2 = 0.5*(-label[i][1] - np.sqrt(np.square(label[i][1]) - 4*label[i][0]*(label[i][2]-CRF_test[i]))) / label[i][0]
#            else:
#                R_2 = 0.5*(-label[i][1]) / label[i][0]
#            bitrate_pre.append(R_2)
#                       
#        # calculate error within 10% & 20% 
#        acc_2 = []
#        acc_1 = []  
#        for i in range(len(R_test)):
#            if np.fabs((np.exp(bitrate_pre[i]) - np.exp(R_test[i])) / np.exp(R_test[i])) <= 0.2:
#                acc_2.append(bitrate_pre[i])
#            if np.fabs((np.exp(bitrate_pre[i]) - np.exp(R_test[i])) / np.exp(R_test[i])) <= 0.1:
#                acc_1.append(bitrate_pre[i])
#        print('TRAIN accuracy less than 20% is ', len(acc_2) / len(R_test))
#        print('TRAIN accuracy less than 10% is ', len(acc_1) / len(R_test))    
#        
##
##
##
##        
##        #val
#        R_test = np.reshape(feature_test[:,3], [len(feature_test), 1])
#        R_test = R_test[int(0.2*len(feature_test))::]
#        #R_test = R_test[int(0.8*len(feature_test))::]
#        CRF_test = np.reshape(feature_test[:, -1], [len(feature_test), 1])
#        CRF_test = CRF_test[int(0.2*len(feature_test))::]
#        #CRF_test = CRF_test[int(0.8*len(feature_test))::]
#        
#        # generate prediction value for all CRF values
#        prediction_tem = prediction_abc_val
#        prediction_abc = []
#        for i in range(len(prediction_tem)):
#            prediction_temt = 33*[prediction_tem[i]]
#            prediction_abc = prediction_abc + prediction_temt
#        prediction_abc = np.array(prediction_abc)   
#        best_acc_1 = 0
#        prediction_abc = ss_y.inverse_transform(prediction_abc)    
#        label = prediction_abc
#        
#        # generate bitrate bascis on predicted model parameters
#        R_1_t = []
#        R_2_t = []
#        bitrate_pre = []
#        for i in range(len(CRF_test)):
#            if np.square(label[i][1]) - 4*label[i][0]*(label[i][2]-CRF_test[i]) > 0:
#                R_2 = 0.5*(-label[i][1] - np.sqrt(np.square(label[i][1]) - 4*label[i][0]*(label[i][2]-CRF_test[i]))) / label[i][0]
#            else:
#                R_2 = 0.5*(-label[i][1]) / label[i][0]
#            bitrate_pre.append(R_2)
#                       
#        # calculate error within 10% & 20% 
#        acc_2 = []
#        acc_1 = []  
#        for i in range(len(R_test)):
#            if np.fabs((np.exp(bitrate_pre[i]) - np.exp(R_test[i])) / np.exp(R_test[i])) <= 0.2:
#                acc_2.append(bitrate_pre[i])
#            if np.fabs((np.exp(bitrate_pre[i]) - np.exp(R_test[i])) / np.exp(R_test[i])) <= 0.1:
#                acc_1.append(bitrate_pre[i])
#        print('VAL accuracy less than 20% is ', len(acc_2) / len(R_test))
#        print('VAL accuracy less than 10% is ', len(acc_1) / len(R_test)) 
#



 
       # generate prediction value for all CRF values
        R_test   = np.reshape(feature720_test[:, 3], [len(feature720_test), 1])
        CRF_test = np.reshape(feature720_test[:, -1], [len(feature720_test), 1])
        prediction_tem = prediction_abc_test
        prediction_abc = []
        for i in range(len(prediction_tem)):
            prediction_temt = 33*[prediction_tem[i]]
            prediction_abc = prediction_abc + prediction_temt
        prediction_abc = np.array(prediction_abc)   
        
        prediction_abc_v = ss_y.inverse_transform(prediction_abc)    
        label = prediction_abc_v
        
        
        # generate bitrate bascis on predicted model parameters
        R_1_t = []
        R_2_t = []
        bitrate_pre = []
        for i in range(len(CRF_test)):
            if np.square(label[i][1]) - 4*label[i][0]*(label[i][2]-CRF_test[i]) > 0:
                R_2 = 0.5*(-label[i][1] - np.sqrt(np.square(label[i][1]) - 4*label[i][0]*(label[i][2]-CRF_test[i]))) / label[i][0]
            else:
                R_2 = 0.5*(-label[i][1]) / label[i][0]
            bitrate_pre.append(R_2)
                       
        # calculate error within 10% & 20% 
        acc_2 = []
        acc_1 = []  
        for i in range(len(R_test)):
            if np.fabs((np.exp(bitrate_pre[i]) - np.exp(R_test[i])) / np.exp(R_test[i])) <= 0.2:
                acc_2.append(bitrate_pre[i])
            if np.fabs((np.exp(bitrate_pre[i]) - np.exp(R_test[i])) / np.exp(R_test[i])) <= 0.1:
                acc_1.append(bitrate_pre[i])
        a_2 = len(acc_2) / len(R_test)
        a_1 = len(acc_1) / len(R_test)
        print('TEST accuracy less than 20% is ', a_2)
        print('TEST accuracy less than 10% is ', a_1) 
        
        if best_acc_1 < a_1:
            best_acc_1 = a_1
            best_acc_2 = a_2
            best_iter = iteration
            saver.save(sess=sess, save_path=save_path)

        print('BEST intera =', best_iter, 'BEST accuracy less than 20% is ', best_acc_2, 'BEST accuracy less than 10% is ', best_acc_1)

sio.savemat('1st.mat', {'Wx_plus_b':Wx_plus_b})
sio.savemat('saveddata11.mat', {'mean_1st':mean1})
sio.savemat('1st_bn.mat', {'Wx_plus_b_bn':Wx_plus_b_bn})
sio.savemat('saveddata12.mat', {'var_1st':var1})
#sio.savemat('saveddata13.mat', {'mean_2st':mean2})
#sio.savemat('saveddata14.mat', {'var_2st':var2})
