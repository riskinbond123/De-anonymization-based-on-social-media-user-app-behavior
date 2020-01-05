# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 22:25:41 2019

@author: risha
"""
import numpy as np
import pandas as pd
#import heapq
#from collections import Counter

data1 = pd.read_csv('output_lcc_graph.txt', sep = " ", header = None)
data2 = pd.read_csv('lcc_user_app.txt', sep = ";", header = None) # Reading the original file
names = ['user','app']
data2.columns = names
modification1 = data2['app'].str.replace(':',' ')


col1 = data1.values[:,[0]] # users

mapped_matrix = []

old_node_matrix = np.unique(col1) # gets unique user nodes

for i in range (1,18950): 
    mapped_matrix.append(i)


dict1 = {}
for i in range (0,18949):
    dict1[(old_node_matrix[i])] = str((mapped_matrix[i]))

user_mapped = data2['user'].map(dict1)

unique_array1 = []

for j in range(0,18949):
    temp=0
    for i in range(0, len(modification1.loc[j].split(" ")),2):
        temp =temp+2
        if(int(modification1.loc[j].split(" ")[i])> 5000):
            unique_array1.append(modification1.loc[j].split(" ")[:temp-2])
            break
        if(i == len(modification1.loc[j].split(" "))-2 ):
            unique_array1.append([])
            break
    print(j)


#count = Counter(unique_array)  #first 6 are the app ratings

#top_apps = heapq.nlargest(5007, count.keys(), key=count.get) # did this to get app users
"""
unique_array = set(unique_array)
unique_array = list(map(float, unique_array))
unique_array.sort()
print(unique_array)
"""
mapped_matrix2 = []

#old_node_matrix2 = unique_array1[6:] # making this variable significantly improved my mapped_matrix construction. Before I was calling the np.uniqu directly in the for loop and it was running very slowly.

for i in range (0, 5000):
    mapped_matrix2.append(i+18949)


dict2 = {}
for i in range (0,5000):
    dict2[i+1] = str((mapped_matrix2[i]))

modification3 =  pd.Series((v for v in unique_array1))

# there are 18928 users with ratings to apps which are in top 5000
max1 = 0
for i in range(0,18949):
    if(len(modification3[i]) >max1):
        max1 = len(modification3[i])
#max1 is 1192
app_matrix = np.zeros([18949,max1])
for i in range (0,18949):
    for j in range (0,len(modification3[i])):
        app_matrix[i][j] = modification3[i][j]
    print(i)    
app_matrix_dataframe = pd.DataFrame(app_matrix)

new_matrix_dataframe = user_mapped

new_dataframe = pd.DataFrame(columns = ['user'])
new_dataframe['user'] = user_mapped


dict3 = {}
dict3[1] = 1
dict3[0.8] = 0.8
dict3[0.6] = 0.6
dict3[0.4] = 0.4
dict3[0.2] = 0.2
dict3[0.9] = 0.8
dict3[0.7] = 0.6
dict3[0.5] = 0.4
dict3[0.3] = 0.2
dict3[0.1] = 0.1

for i in range (0,max1-1):
    if i % 2 == 0:
        new_dataframe[i] = app_matrix_dataframe[i].map(dict2)
    else:
        new_dataframe[i] = app_matrix_dataframe[i].map(dict3)
    print (i)

final_dataset_userApp = pd.DataFrame(columns = ['user', 'app', 'rating'])
for i in range(0,18949):
    for j in range(1, max1-2, 2):        
        if(new_dataframe.iloc[i][j] <= 1):
            final_dataset_userApp.loc[len(final_dataset_userApp.index)] = pd.Series({'user':new_dataframe['user'][i], 'app': new_dataframe.iloc[i][j-1], 'rating': new_dataframe.iloc[i][j]})
        else:
            break
#        final_dataset_userApp.append(pd.Series({'user':new_dataframe['user'][i], 'app': new_dataframe[j-1][i], 'rating': new_dataframe[j][i]}), ignore_index=True)
    print(i)
#print(unique_array.size)