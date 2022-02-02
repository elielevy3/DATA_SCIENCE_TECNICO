#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 14:52:17 2022

@author: elie
"""

#################### SCALING ####################

import os
os.chdir('/home/elie/Documents/Tecnico/2ND_PERIOD/DS/PROJECT/CODE/')
   

from pandas import read_csv, DataFrame, concat, unique
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import subplots, show, figure, savefig
from ds_charts import get_variable_types, multiple_line_chart, plot_evaluation_results
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from numpy import ndarray
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


register_matplotlib_converters()


############################# GET DATA ###############################
path = "Data/"
file = path+"air_quality_tabular_without_na"
#file = path+"nyc_car_crash_without_na"
filename = file+".csv"
data = read_csv(filename, na_values='', parse_dates=True, infer_datetime_format=True)


# SPLIT DATA BASED ON TYPE OF VARIABLE
variable_types = get_variable_types(data)
numeric_vars = variable_types['Numeric']
symbolic_vars = variable_types['Symbolic']
boolean_vars = variable_types['Binary']

df_nr = data[numeric_vars]
df_sb = data[symbolic_vars]
df_bool = data[boolean_vars]

# remove symbolic values before computation : date, time, id
data = data.drop(symbolic_vars, axis=1)

############################# NORMALIZATION ###############################


# Z SCORE
transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
norm_data_zscore = concat([tmp, df_sb,  df_bool], axis=1)
#norm_data_zscore.to_csv(f'{file}_scaled_zscore.csv', index=False)
print(norm_data_zscore.describe())

norm_data_zscore = norm_data_zscore.drop(symbolic_vars, axis=1)


# MIN MAX SCALER
transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
norm_data_minmax = concat([tmp, df_sb,  df_bool], axis=1)
#norm_data_minmax.to_csv(f'{file}_scaled_minmax.csv', index=False)
print(norm_data_minmax.describe())

norm_data_minmax = norm_data_minmax.drop(symbolic_vars, axis=1)



# fig, axs = subplots(1, 3, figsize=(20,10),squeeze=False)
# axs[0, 0].set_title('Original data')
# data.boxplot(ax=axs[0, 0])
# axs[0, 1].set_title('Z-score normalization')
# norm_data_zscore.boxplot(ax=axs[0, 1])
# axs[0, 2].set_title('MinMax normalization')
# norm_data_minmax.boxplot(ax=axs[0, 2])
# show()


################################## KNN ##################################

nb_rows = norm_data_zscore.shape[0]
sample_pct = 0.33
norm_data_zscore = norm_data_zscore.sample(n=round(nb_rows*sample_pct), random_state=1)
norm_data_minmax = norm_data_minmax.sample(n=round(nb_rows*sample_pct), random_state=1)

potential_cols = ["ALARM"]
nvalues = [10, 15, 20, 25, 30, 35, 40, 45]
dist = ['manhattan', 'euclidean', 'chebyshev']
values = {}
best = (0, '')
last_best = 0

for c in potential_cols:
    target = c
    y = norm_data_zscore.pop(target).values
    X_train, X_test, y_train, y_test = train_test_split(norm_data_zscore, y, test_size=0.33, random_state=42)
    labels = unique(y_train)
    labels.sort()
    for d in dist:
        yvalues = []
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            current_accuracy = accuracy_score(y_test, y_pred) 
            yvalues.append(current_accuracy)
            print("For column : "+str(c)+" Accuracy with n = "+str(n)+ " and distance : "+str(d)+" => "+str(current_accuracy))
            # if yvalues[-1] > last_best:
            #     best = (n, d)
            #     last_best = yvalues[-1]
        values[d] = yvalues
    figure()
    multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel='accuracy', percentage=True)
    #savefig('images/{file_tag}_knn_study.png')
    show()

# figure()
# multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel='accuracy', percentage=True)
# #savefig('images/{file_tag}_knn_study.png')
# show()
# print('Best results with %d neighbors and %s'%(best[0], best[1]))

# ###### CONFUSION MATRIX #######

clf = knn = KNeighborsClassifier(n_neighbors=6, metric="manhattan")
clf.fit(X_train, y_train)
train_pred = clf.predict(X_train)
test_pred = clf.predict(X_test)
plot_evaluation_results(labels, y_train, train_pred, y_test, test_pred)
#savefig('images/{file_tag}_knn_best.png')
show()



###############################   TEST  ###################################"


# # GET THE PREPROCESSED DATA WITHOUT NA
# path = "Data/"
# #file = path+"air_quality_tabular_without_na"
# file = path+"NYC_collisions_tabular"
# filename = file+".csv"
# data_bis = read_csv(filename, na_values='', parse_dates=True, infer_datetime_format=True)


# for col in data_bis.columns:    
#     print("COL : "+str(col))
#     print("/////////////////")
#     print(data_bis[col].value_counts())
#     print("\n\n\n")

