# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 15:54:46 2019

@author: rishab kinnerkar
"""
import numpy as np
import pandas as pd
data = pd.read_csv('output_graph.txt', sep = " ", header = None) # Reading the original file

#data = data.astype(np.int)
#print(data.astype(np.int32).dtypes)

col1 = data.values[:,[0]] # Putting the first column in an array
#col2 = data.values[:,[1]] # Putting the second column in an array

#print(np.unique(col1))
# running col1.size and col2.size gives 2235834. These are the total nodes.
# running np.unique(col1).size and np.unique(col2).size gives 259278. These are the unique nodes which are returned sorted.

#print(np.unique(col1)[0])

#col1_series = pd.Series(col1.tolist())
#col2_series = pd.Series(col2.tolist())
#col2_series = pd.Series(col2)

mapped_matrix = []

old_node_matrix = np.unique(col1) # making this variable significantly improved my mapped_matrix construction. Before I was calling the np.uniqu directly in the for loop and it was running very slowly.



for i in range (1,259279):
    mapped_matrix.append(i)



dict1 = {}
for i in range (0,259278):
    dict1[(old_node_matrix[i])] = str((mapped_matrix[i]))

#print(data)
#print(data)
#print(data[0].map(dict1))
#print(data[1].map(dict1))
#print(data)

s = pd.Series(['1']*data[0].map(dict1).size)
print(s)

pd.concat([data[0].map(dict1),data[1].map(dict1),s], axis=1).to_csv('output_graph_postprocessed.txt', sep=' ', index = False, header=False)

#print(col1_series.map(dict1))

#print(type(data))
#print(data.replace(to_replace = old_node_matrix.tolist(), value = mapped_matrix.tolist(), regex = True))

    
#print(data)

#print(col1)
#print(data)
