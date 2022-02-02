#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:36:50 2022

@author: elie
"""

import time
from numpy import ndarray, std, argsort
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig, show
from ds_charts import plot_evaluation_results, multiple_line_chart, horizontal_bar_chart, HEIGHT, get_variable_types
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

######################## GET DATA ########################
path = "Data/"
#file_tag = "air_quality_tabular_without_na"
file_tag = "nyc_car_crash_without_na"
filename = path+file_tag+".csv"
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



### split betweet test and train

# target = "BODILY_INJURY"
target = "COMPLAINT"
y = data.pop(target).values
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.25, random_state=42)
labels = unique(y_train)
labels.sort()


# ######################## MLP #################################


# ### figuring out optimal parameters

# start_time = time.time()
    
# lr_type = ['constant', 'invscaling', 'adaptive']
# max_iter = [100, 500, 1000]
# learning_rate = [0.1, 0.3, 0.7, 0.9]
# best = ('', 0, 0)
# last_best = 0
# best_model = None

# cols = len(lr_type)
# figure()
# fig, axs = subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)
# for k in range(len(lr_type)):
#     d = lr_type[k]
#     values = {}
#     for lr in learning_rate:
#         yvalues = []
#         for n in max_iter:
#             mlp = MLPClassifier(activation='logistic', solver='sgd', learning_rate=d,
#                                 learning_rate_init=lr, max_iter=n, verbose=False)
#             mlp.fit(X_train, y_train)
#             y_pred = mlp.predict(X_test)
#             current_accuracy = accuracy_score(y_test, y_pred)
#             yvalues.append(current_accuracy)
#             print("for lr_type : "+str(lr_type[k])+" , lr : "+str(lr)+" and max_iter="+str(n)+" -> accuracy : "+str(current_accuracy))
#             if yvalues[-1] > last_best:
#                 best = (d, lr, n)
#                 last_best = yvalues[-1]
#                 best_model = mlp
#         values[lr] = yvalues
#     multiple_line_chart(max_iter, values, ax=axs[0, k], title=f'MLP with lr_type={d}',
#                            xlabel='mx iter', ylabel='accuracy', percentage=True)
# #savefig(f'images/{file_tag}_mlp_predict.png')
# show()
# print(f'Best results with lr_type={best[0]}, learning rate={best[1]} and {best[2]} max iter, with accuracy={last_best}')

# print("--- %s seconds ---" % (time.time() - start_time))

# ### plot with optimal parameters

# pred_train = best_model.predict(X_train)
# pred_test = best_model.predict(X_test)
# plot_evaluation_results(labels, y_train, pred_train, y_test, pred_test)

# #savefig(f'images/{file_tag}_mlp_best.png')
# show()

########################### GRADIENT BOOSTING ###########################

### figuring out optimal parameters

start_time = time.time()

n_estimators = [5, 25, 50, 75]
max_depths = [10, 25, 50]
learning_rate = [.1, 0.3, 0.6]
best = ('', 0, 0)
last_best = 0
best_model = None

cols = len(max_depths)
figure()
fig, axs = subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)
for k in range(len(max_depths)):
    d = max_depths[k]
    values = {}
    for lr in learning_rate:
        yvalues = []
        for n in n_estimators:
            gb = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr)
            gb.fit(X_train, y_train)
            y_pred = gb.predict(X_test)
            current_accuracy = accuracy_score(y_test, y_pred) 
            yvalues.append(current_accuracy)
            print("for max depth : "+str(max_depths[k])+" , lr : "+str(lr)+" and n_estimators="+str(n)+" -> accuracy : "+str(current_accuracy))
            if yvalues[-1] > last_best:
                best = (d, lr, n)
                last_best = yvalues[-1]
                best_model = gb
        values[lr] = yvalues
    multiple_line_chart(n_estimators, values, ax=axs[0, k], title=f'Gradient Boorsting with max_depth={d}',
                            xlabel='nr estimators', ylabel='accuracy', percentage=True)
#savefig(f'images/{file_tag}_gb_study.png')
show()
print('Best results with depth=%d, learning rate=%1.2f and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], last_best))
print("--- %s seconds ---" % (time.time() - start_time))

### plot optimal parameters

pred_train = best_model.predict(X_train)
pred_test = best_model.predict(X_test)
plot_evaluation_results(labels, y_train, pred_train, y_test, pred_test)
#savefig(f'images/{file_tag}_gb_best.png')
show()

### feature importance

variables = data.columns
importances = best_model.feature_importances_
indices = argsort(importances)[::-1]
stdevs = std([tree[0].feature_importances_ for tree in best_model.estimators_], axis=0)
elems = []
for f in range(len(variables)):
    elems += [variables[indices[f]]]
    print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')

figure()
horizontal_bar_chart(elems, importances[indices], stdevs[indices], title='Gradient Boosting Features importance', xlabel='importance', ylabel='variables')
#savefig(f'images/{file_tag}_gb_ranking.png')