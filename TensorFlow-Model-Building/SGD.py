# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 19:09:54 2019

@author: risha
"""
import pandas as pd
import tensorflow as tf
import numpy as np

X = 'user-graph3.txt'
Y = 'app-user3.txt'
data1 = pd.read_csv(X, sep='\t', header=None)
data2 = pd.read_csv(Y, sep='\t', header=None)
data3 = pd.read_csv('Combined3.txt', sep='\t', header=None)

def split(data):
    # control randomization for reproducibility
    np.random.seed(42)
    np.random.seed(42)
    train, test = tf.model_selection.train_test_split(data)
    x_train = train.loc[:, train.columns != 'chd']
    y_train = train['chd']
    x_test = test.loc[:, test.columns != 'chd']
    y_test = test['chd']
    return x_train, y_train, x_test, y_test
inner_product_data = pd.DataFrame(np.zeros((18928, 121)))
for j in range(1, 121):
    print(j)
    for i in range(0,18928):
        inner_product_data[j][i] = data1[j][i]*data2[j][i]

inner_product_data.to_csv('inner_product_data.txt', sep=' ', index = False, header=False)