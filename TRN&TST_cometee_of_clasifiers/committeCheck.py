#!/usr/bin/env python
# coding: utf-8

# CLASSIFIERS DEFINITION
from sklearn.preprocessing import StandardScaler
import sklearn
import sklearn.metrics as metrics
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
# DATA PREPROCESSING
import urllib
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# url = "http://wmii.uwm.edu.pl/~artem/data/heart_df.dat"
# print(url)
# file = urllib.request.urlopen(url)
# for line in file:
#     decoded_line = line.decode("utf-8")
#     for i in range(len(decoded_line)):
#         if decoded_line[i] == ',':
#             csv_file = True
#             break
#
# # extracting the file extension and the name of the decision system,,
# temp = url
# extension = ""
# dataset_name = ""
# przelacznik = 0
#
# for i in range(len(temp) - 1, -1, -1):
#     if temp[i] != '.' and przelacznik == 0:
#         extension = temp[i] + extension
#     if temp[i] == '.':
#         przelacznik = 1
#     if przelacznik == 1 and temp[i] != '/' and temp[i] != '.':
#         dataset_name = temp[i] + dataset_name
#     if temp[i] == '/':
#         break
#
# csv_file = False
#
# # Temporary reading to detect number of objects and attributes,
# if csv_file:
#     dataset = pd.read_csv(url, sep=',', dtype=str)
# else:
#     dataset = pd.read_csv(url, sep=' ', dtype=str)
#
dataset = pd.read_csv('wine.data', sep=',', dtype=str)
print(dataset)
attr_no = len(dataset.iloc[1])
print(attr_no)
dec_index = attr_no - 1
obj_no = len(dataset)

names = []
for i in range(1, attr_no): names.append('a' + str(i))
names.append('class')
#
# if csv_file:
#     dataset = pd.read_csv(url, sep=',', dtype=str)
# else:
#     dataset = pd.read_csv(url, sep=' ', dtype=str)
#
print(dataset.columns[1])
print(dataset.columns)

# print(dataset)
# print('========================')
# dataset = pd.read_csv('wine.data', sep=' ', dtype=str)
# print(dataset)


# print('========================')
# print(dataset1)
# print('========================')


# dataset.head()
classes = np.unique(dataset.to_numpy()[:, attr_no - 14])
print(classes)
class_size_orig = []
for i in range(0, len(classes)):
    class_size_orig.append(sum(dataset.iloc[:, dec_index] == classes[i]))
minimal_class_size = min(class_size_orig)

print("minimal_class_size = ", minimal_class_size)
print("names", names)
dataset_list = dataset.values.tolist()
print(type(dataset_list))
print(dataset_list[0])
# print("dataset_name = ", dataset_name)

# BASIC SPLIT TO TRN AND TST
train, test = train_test_split(dataset, test_size=0.2, random_state=0)
train_list = train.values.tolist()
y_train = train['class']
y_test = test['class']
X_train = train.drop(columns=['class'])
X_test = test.drop(columns=['class'])


def bin_cm_params(cm, y_pred):
    #     print('cm = ',cm)
    tp = cm[0][0]
    fn = cm[0][1]
    fp = cm[1][0]
    tn = cm[1][1]
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracyP = tp / (tp + fn)
    accuracyN = tn / (tn + fp)
    balanced_accuracy = (accuracyP + accuracyN) / 2.
    return round(accuracy, 3), round(precision, 3), round(recall, 3), round(accuracyP, 3), round(accuracyN, 3), round(
        balanced_accuracy, 3)


def committees_of_classifiers(classifiers_result):
    result = 0
    no_of_classifiers = 0
    for i in range(len(classifiers_result)):
        result += classifiers_result[i][1]
        no_of_classifiers += 1
    return result/no_of_classifiers


def classifiers_set(X_train, y_train, X_test, y_test):
    classification_result = []

    # kNNClassifier(canberra)
    classifier = KNeighborsClassifier(metric="canberra", n_neighbors=i)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    accuracy, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred)
    classification_result.append([str(i) + 'canberra', balanced_accuracy])

    # GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    accuracy, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred)
    classification_result.append(['GaussianNB', balanced_accuracy])

    # DecisionTreeClassifier
    classifier = sklearn.tree.DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=2)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    accuracy, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred)
    classification_result.append(['DecisionTreeClassifier', balanced_accuracy])

    # RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=10, max_depth=4, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    accuracy, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred)
    classification_result.append(['RandomForestClassifier', balanced_accuracy])

    # SVM rbf
    classifier = sklearn.svm.SVC(kernel='rbf', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    accuracy, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred)
    classification_result.append(['SVM_rbf', balanced_accuracy])

    print('done')

    return classification_result


#     # KLASYFIKACJA BAZOWA za pomocÄ TRN
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
classifiers_set_result = classifiers_set(X_train, y_train, X_test, y_test)
print('Result for original TRN and TST')
print(classifiers_set_result[0][1])
print(committees_of_classifiers(classifiers_set_result))

