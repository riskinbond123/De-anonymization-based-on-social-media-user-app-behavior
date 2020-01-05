# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 22:25:41 2019

@author: risha
"""
import numpy as np
import pandas as pd

data1 = pd.read_csv('output_graph.txt', sep = " ", header = None)
data2 = pd.read_csv('lcc_user_app.txt', sep = ";", header = None) # Reading the original file
names = ['user','app']
data2.columns = names
modification1 = data2['app'].str.replace(':',' ')


col1 = data1.values[:,[0]]

mapped_matrix = []

old_node_matrix = np.unique(col1) # making this variable significantly improved my mapped_matrix construction. Before I was calling the np.uniqu directly in the for loop and it was running very slowly.

for i in range (1,259279):
    mapped_matrix.append(i)


dict1 = {}
for i in range (0,259278):
    dict1[(old_node_matrix[i])] = str((mapped_matrix[i]))

user_mapped = data2['user'].map(dict1)

unique_array = []
for j in range(0,259278):
    row1 = modification1.loc[j]
    #print([float(i) for i in row1.split(' ')])
    unique_array.extend(list(set([float(i) for i in row1.split(' ')])))

unique_array = set(unique_array)
unique_array = list(map(float, unique_array))
unique_array.sort()
#print(unique_array)

mapped_matrix2 = []

old_node_matrix2 = unique_array[7:] # making this variable significantly improved my mapped_matrix construction. Before I was calling the np.uniqu directly in the for loop and it was running very slowly.

for i in range (0,(170910-7)):
    mapped_matrix2.append(i+259279)


dict2 = {}
for i in range (0,(170910-7)):
    dict2[(int)(old_node_matrix2[i])] = str((mapped_matrix2[i]))

modification3 =  (data2['app'].str.replace(':',' ')).str.split(' ')

app_matrix = np.zeros([259278,1192])
for i in range (0,259278):
    for j in range (0,len(modification3[i])):
        app_matrix[i][j] = modification3[i][j]
    print(i)    
app_matrix_dataframe = pd.DataFrame(app_matrix)

new_matrix_dataframe = user_mapped

new_dataframe = pd.DataFrame(columns = ['user'])
new_dataframe['user'] = user_mapped

for i in range (0,1191):
    if i % 2 == 0:
        new_dataframe['i'] = app_matrix_dataframe[i].map(dict2)
    else:
        new_dataframe['i'] = app_matrix_dataframe[i]
    print (i)

final_dataset_userApp = pd.DataFrame(columns = ['user', 'app', 'rating'])
for i in range(0,259278):
    for j in range(1, 1190, 2):
        if(new_dataframe[j][i] == 0.0):
            break
        if(new_dataframe[j][i] == 0.9 or new_dataframe[j][i] == 0.7 or new_dataframe[j][i] == 0.5 or new_dataframe[j][i] == 0.3 or new_dataframe[j][i] == 0.9):
            new_dataframe[j][i] = new_dataframe[j][i] - 0.1    
        final_dataset_userApp.loc[len(final_dataset_userApp.index)] = pd.Series({'user':new_dataframe['user'][i], 'app': new_dataframe[j-1][i], 'rating': new_dataframe[j][i]})
    print(i)
#print(unique_array.size)
