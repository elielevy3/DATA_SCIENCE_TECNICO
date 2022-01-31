#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 09:34:39 2022

@author: elie
"""

import os
os.chdir('/home/elie/Documents/Tecnico/2ND_PERIOD/DS/PROJECT/CODE/')

from numpy import ndarray
from imblearn.over_sampling import SMOTE
from matplotlib.pyplot import figure, savefig, show
from pandas import read_csv, concat, DataFrame, Series, unique
from ds_charts import bar_chart, multiple_bar_chart, get_variable_types, plot_evaluation_results
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB



### GETTING DATA
path = "Data/"
# file_tag = "air_quality_tabular_without_na"
file_tag = "nyc_car_crash_without_na"
filename = path+file_tag+".csv"
original = read_csv(filename, na_values='', parse_dates=True, infer_datetime_format=True)
# class_var = "BODILY_INJURY"
class_var = "COMPLAINT"
# class_var = "ALARM"


# SPLIT DATA BASED ON TYPE OF VARIABLE
variable_types = get_variable_types(original)
numeric_vars = variable_types['Numeric']
symbolic_vars = variable_types['Symbolic']
boolean_vars = variable_types['Binary']

df_nr = original[numeric_vars]
df_sb = original[symbolic_vars]
df_bool = original[boolean_vars]

# remove symbolic values before computation : date, time, id
original = original.drop(symbolic_vars, axis=1)


########################### DATA BALANCING ###########################


### CHECK HOW UNBALANCED DATA IS RIGHT NOW
target_count = original[class_var].value_counts()
positive_class = target_count.idxmin()
negative_class = target_count.idxmax()
print('Minority class=', positive_class, ':', target_count[positive_class])
print('Majority class=', negative_class, ':', target_count[negative_class])
print('Proportion:', round(target_count[positive_class] / target_count[negative_class], 2), ': 1')
values = {'Original': [target_count[positive_class], target_count[negative_class]]}

figure()
bar_chart(target_count.index, target_count.values, title='Class balance')
show()

df_positives = original[original[class_var] == positive_class]
df_negatives = original[original[class_var] == negative_class]


### UNDERSAMPLING
df_neg_sample = DataFrame(df_negatives.sample(len(df_positives)))
df_under = concat([df_positives, df_neg_sample], axis=0)
#df_under.to_csv(f'data/{file_tag}_under.csv', index=False)
values['UnderSample'] = [len(df_positives), len(df_neg_sample)]
print('Minority class=', positive_class, ':', len(df_positives))
print('Majority class=', negative_class, ':', len(df_neg_sample))
print('Proportion:', round(len(df_positives) / len(df_neg_sample), 2), ': 1')


# ### OVERSAMPLING 
# df_pos_sample = DataFrame(df_positives.sample(len(df_negatives), replace=True))
# df_over = concat([df_pos_sample, df_negatives], axis=0)
# #df_over.to_csv(f'Data/{file_tag}_over.csv', index=False)
# values['OverSample'] = [len(df_pos_sample), len(df_negatives)]
# print("With OVERSAMPLING")
# print('Minority class=', positive_class, ':', len(df_pos_sample))
# print('Majority class=', negative_class, ':', len(df_negatives))
# print('Proportion:', round(len(df_pos_sample) / len(df_negatives), 2), ': 1')


### SMOTE
# RANDOM_STATE = 42

# smote = SMOTE(sampling_strategy='minority', random_state=RANDOM_STATE)
# y = original.pop(class_var).values
# X = original.values
# smote_X, smote_y = smote.fit_resample(X, y)
# df_smote = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
# df_smote.columns = list(original.columns) + [class_var]
# #df_smote.to_csv(f'Data/{file_tag}_smote.csv', index=False)

# smote_target_count = Series(smote_y).value_counts()
# values['SMOTE'] = [smote_target_count[positive_class], smote_target_count[negative_class]]
# print("With SMOTE")
# print('Minority class=', positive_class, ':', smote_target_count[positive_class])
# print('Majority class=', negative_class, ':', smote_target_count[negative_class])
# print('Proportion:', round(smote_target_count[positive_class] / smote_target_count[negative_class], 2), ': 1')

# figure()
# multiple_bar_chart([positive_class, negative_class], values, title='Target', xlabel='frequency', ylabel='Class balance')
# show()


#################################### NB ####################################


# we need to make sure there is only positive values in the input dataset (X)

# get columns where there are negative values: 
cols_with_neg = original.columns[(original < 0).any()].tolist()

# add 1 everywhere
for col in cols_with_neg:
    original[col] = original[col] + 1
    
# check if there is still columns with negative values
cols_with_neg = original.columns[(original < 0).any()].tolist()

# remove rows that contain negative value in this column
for col in cols_with_neg: 
    indexNames = original[ original['PERSON_AGE'] < 0 ].index    
    original.drop(indexNames , inplace=True)

print(original.columns[(original < 0).any()].tolist())


y = original.pop(class_var).values
X_train, X_test, y_train, y_test = train_test_split(original, y, test_size=0.25, random_state=42)
labels = unique(y_train)
labels.sort()

# clf = GaussianNB()
# clf.fit(X_train, y_train)
# train_pred = clf.predict(X_train)
# test_pred = clf.predict(X_test)
#plot_evaluation_results(labels, y_train, train_pred, y_test, test_pred)

# savefig('images/{file_tag}_nb_best.png')
show()

estimators = {'GaussianNB': GaussianNB(),
              'MultinomialNB': MultinomialNB(),
              'BernoulliNB': BernoulliNB()
              #'CategoricalNB': CategoricalNB
              }

xvalues = []
yvalues = []
for clf in estimators:
    xvalues.append(clf)
    estimators[clf].fit(X_train, y_train)
    y_pred = estimators[clf].predict(X_test)
    yvalues.append(accuracy_score(y_test, y_pred))

figure()
bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
#savefig(f'images/{file_tag}_nb_study.png')
show()


