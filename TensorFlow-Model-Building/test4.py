# -- coding: utf-8 --
"""
Created on Sat Mar 30 22:40:48 2019

@author: Administrator
"""
from __future__ import print_function
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import combinations
import random

x_train_data = np.loadtxt('x_train(1-500)-user.txt')
y_train_data = np.loadtxt('y_train(1-500)-app.txt')
x_test_data = np.loadtxt('x_test(501-1000)-user.txt')
y_test_data = np.loadtxt('y_test(501-1000)-app.txt')

# Get all combinations of length 2 and lable them to 0
combins_train = [c for c in  combinations(range(1,501), 2)]
combins_train = np.asarray(combins_train)
combins_test = [c for c in  combinations(range(501,1001), 2)]
combins_test = np.asarray(combins_test)







label_train = np.zeros(124750) #comb(500, 2)
label_test = np.zeros(124750)

sample_train = np.c_[combins_train,label_train]
sample_test = np.c_[combins_test,label_test]

# append the possitive samples
for i in range(1,501):
    sample_train = np.row_stack((sample_train,[i,i,1]))
    
for i in range(501,1001):
    sample_test = np.row_stack((sample_test,[i,i,1]))

# convert Numpy array to Panda DataFrame
sample_train = pd.DataFrame(sample_train)
sample_test = pd.DataFrame(sample_test)
x_train_data = pd.DataFrame(x_train_data)
y_train_data = pd.DataFrame(y_train_data)
x_test_data = pd.DataFrame(x_test_data)
y_test_data = pd.DataFrame(y_test_data)

# get the table of train
table_train = pd.merge(sample_train, x_train_data, how='left', left_on=0,right_on=0)
table_train = pd.merge(table_train, y_train_data, how='left', left_on='1_x',right_on=0)
table_train = table_train.drop('0_y',axis=1)
table_train= table_train.dropna()

# convert Panda DataFrame to Numpy array
table_train =  np.asarray(table_train)

# get the table of test(May not be needed)
table_test = pd.merge(sample_test, x_test_data, how='left', left_on=0,right_on=0)
table_test = pd.merge(table_test, y_test_data, how='left', left_on='1_x',right_on=0)
table_test = table_test.drop('0_y',axis=1)
table_test= table_test.dropna()

# convert Panda DataFrame to Numpy array
table_test =  np.asarray(table_test)

# get x_train_1 and y_train_1 and label_train
x_train_1 = table_train[:,3:123]
y_train_1 = table_train[:,123:243]
label_train = table_train[:,2]

# get x_test_1 and y_test_1 (May not be needed)
x_test_1 = table_test[:,3:123]
y_test_1 = table_test[:,123:243]
label_test = table_test[:,2]

# get x_train(cross product by x_train_1 and y_train_1) and y_train
x_train = np.zeros([x_train_1.shape[0],120]) #[125250,120]
y_train = np.zeros([x_train_1.shape[0],2]) #[125250,2]
for i in np.arange(x_train_1.shape[0]):
    x_train[i,:]=(x_train_1[i,:]*y_train_1[i,:])
    if label_train[i] == 1:
        y_train[i] = [0,1]
    else:
        y_train[i] = [1,0]


# get x_test(cross product by x_test_1 and y_test_1) and y_test  (May not be needed)
x_test = np.zeros([x_test_1.shape[0],120])
y_test = np.zeros([x_test_1.shape[0],2])
for i in np.arange(x_test_1.shape[0]):
    x_test[i,:]=(x_test_1[i,:]*y_test_1[i,:])
    if label_test[i] == 1:
        y_test[i] = [0,1]
    else:
        y_test[i] = [1,0]       
 
x_test_data =  np.asarray(x_test_data)
y_test_data =  np.asarray(y_test_data)

# Parameters
learning_rate = 10
training_epochs = 25
batch_size = 64
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 120]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 2]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([120,2]))
b = tf.Variable(tf.zeros([2]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training cycle
    for epoch in range(20):
        avg_cost = 0.
        total_batch = int(x_train.shape[0]/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = x_train[int(i*batch_size):min((i+1)*batch_size,x_train.shape[0])]
            batch_ys = y_train[int(i*batch_size):min((i+1)*batch_size,x_train.shape[0])]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    test_sample_temp=np.zeros([x_test_data.shape[0],120],dtype=np.float)
    test_label_temp=np.zeros([x_test_data.shape[0],2],dtype=np.int)
    count_accuracy=0.0
    for i in np.arange(x_test_data.shape[0]):
        for j in np.arange(x_test_data.shape[0]):
               test_sample_temp[j,:]=x_test_data[i,1:]*y_test_data[j,1:]
        pred_value=sess.run(pred,feed_dict={x:test_sample_temp})
        print(np.argmax(pred_value[:,1]),i)
        if np.argmax(pred_value[:,1])==i:
            count_accuracy+=1.0
            
    print("accuracy: {}".format(count_accuracy/x_test_data.shape[0]))