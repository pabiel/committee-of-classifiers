#!/usr/bin/env python
# coding: utf-8

# In[4]:


# DATA PREPROCESSING
import urllib
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
url = "http://wmii.uwm.edu.pl/~artem/data/heart_df.dat"
file = urllib.request.urlopen(url)
for line in file:
    decoded_line = line.decode("utf-8")
    for i in range(len(decoded_line)):
        if (decoded_line[i] == ','):
            csv_file = True
            break

#extracting the file extension and the name of the decision system,,
temp = url
extension = ""
dataset_name = ""
przelacznik = 0
for i in range(len(temp) - 1, -1, -1):
    if temp[i] != '.' and przelacznik == 0:
        extension = temp[i] + extension
    if temp[i] == '.':
        przelacznik = 1
    if przelacznik == 1 and temp[i] != '/' and temp[i] != '.':
        dataset_name = temp[i] + dataset_name
    if temp[i] == '/':
        break

csv_file=False

# Temporary reading to detect number of objects and attributes,
if(csv_file==True):
    dataset = pd.read_csv(url, sep=',', dtype=str)
else:
    dataset = pd.read_csv(url, sep=' ', dtype=str)

attr_no = len(dataset.iloc[1])
dec_index = attr_no - 1
obj_no = len(dataset)

names = []
for i in range(1, attr_no): names.append('a' + str(i))
names.append('class')

if(csv_file==True):
    dataset = pd.read_csv(url, sep=',', dtype=str)
else:
    dataset = pd.read_csv(url, sep=' ', dtype=str)

print(dataset.columns[1])
print(dataset.columns)

# dataset.head()
classes = np.unique(dataset.to_numpy()[:, attr_no - 1])
class_size_orig = []
for i in range(0, len(classes)):
    class_size_orig.append(sum(dataset.iloc[:, dec_index] == classes[i]))
minimal_class_size = min(class_size_orig)

print("minimal_class_size = ",minimal_class_size)
print("names",names)
dataset_list = dataset.values.tolist()
print(type(dataset_list))
print(dataset_list[0])
print("dataset_name = ",dataset_name)

# BASIC SPLIT TO TRN AND TST
train, test = train_test_split(dataset, test_size=0.2)
train_list=train.values.tolist()
y_train=train['class']
y_test=test['class']
X_train=train.drop(columns=['class'])
X_test=test.drop(columns=['class'])


# In[5]:


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
def bin_cm_params(cm,y_pred):
#     print('cm = ',cm)
    tp=cm[0][0]
    fn=cm[0][1]
    fp=cm[1][0]
    tn=cm[1][1]
    accuracy = metrics.accuracy_score(y_test,y_pred)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    accuracyP=tp/(tp+fn)
    accuracyN=tn/(tn+fp)
    balanced_accuracy=(accuracyP+accuracyN)/2.
    return round(accuracy,3),round(precision,3),round(recall,3),round(accuracyP,3),round(accuracyN,3),round(balanced_accuracy,3)

def classifiers_set(X_train,y_train,X_test,y_test):
    classification_result=[]
    #DummyClassifier
    classifier = DummyClassifier(strategy="most_frequent")
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test,y_pred)
    accuracy,precision,recall,accuracyP,accuracyN,balanced_accuracy = bin_cm_params(cm,y_pred)
    classification_result.append(['DummyClassifier',balanced_accuracy])

    #kNNClassifier
    for i in range(1,2):
        classifier = KNeighborsClassifier(metric="euclidean",n_neighbors=i)
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        cm = metrics.confusion_matrix(y_test,y_pred)
        accuracy,precision,recall,accuracyP,accuracyN,balanced_accuracy = bin_cm_params(cm,y_pred)
        classification_result.append([str(i)+'NN_euclidean',balanced_accuracy])
        classifier = KNeighborsClassifier(metric="manhattan",n_neighbors=i)
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        cm = metrics.confusion_matrix(y_test,y_pred)
        accuracy,precision,recall,accuracyP,accuracyN,balanced_accuracy = bin_cm_params(cm,y_pred)
        classification_result.append([str(i)+'NN_manhattan',balanced_accuracy])
        classifier = KNeighborsClassifier(metric="cityblock",n_neighbors=i)
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        cm = metrics.confusion_matrix(y_test,y_pred)
        accuracy,precision,recall,accuracyP,accuracyN,balanced_accuracy = bin_cm_params(cm,y_pred)
        classification_result.append([str(i)+'NN_cityblock',balanced_accuracy])
        classifier = KNeighborsClassifier(metric="cosine",n_neighbors=i)
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        cm = metrics.confusion_matrix(y_test,y_pred)
        accuracy,precision,recall,accuracyP,accuracyN,balanced_accuracy = bin_cm_params(cm,y_pred)
        classification_result.append([str(i)+'NN_cosine',balanced_accuracy])
        classifier = KNeighborsClassifier(metric="l1",n_neighbors=i)
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        cm = metrics.confusion_matrix(y_test,y_pred)
        accuracy,precision,recall,accuracyP,accuracyN,balanced_accuracy = bin_cm_params(cm,y_pred)
        classification_result.append([str(i)+'NN_l1',balanced_accuracy])
        classifier = KNeighborsClassifier(metric="l2",n_neighbors=i)
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        cm = metrics.confusion_matrix(y_test,y_pred)
        accuracy,precision,recall,accuracyP,accuracyN,balanced_accuracy = bin_cm_params(cm,y_pred)
        classification_result.append([str(i)+'NN_l2',balanced_accuracy])
        classifier = KNeighborsClassifier(metric="nan_euclidean",n_neighbors=i)
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        cm = metrics.confusion_matrix(y_test,y_pred)
        accuracy,precision,recall,accuracyP,accuracyN,balanced_accuracy = bin_cm_params(cm,y_pred)
        classification_result.append([str(i)+'NN_nan_euclidean',balanced_accuracy])
    #DecisionTreeClassifier
    classifier = sklearn.tree.DecisionTreeClassifier(criterion = 'entropy', random_state = 100, max_depth=2)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test,y_pred)
    accuracy,precision,recall,accuracyP,accuracyN,balanced_accuracy = bin_cm_params(cm,y_pred)
    classification_result.append(['DecisionTreeClassifier',balanced_accuracy])
    # #XGBClassifier
    # classifier = XGBClassifier()
    # classifier.fit(X_train,y_train)
    # use_label_encoder=False
    # y_pred = classifier.predict(X_test)
    # cm = metrics.confusion_matrix(y_test,y_pred)
    # accuracy,precision,recall,accuracyP,accuracyN,balanced_accuracy = bin_cm_params(cm,y_pred)
    # classification_result.append(['XGBClassifier',balanced_accuracy])
    #RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=10,max_depth=4,criterion='entropy',random_state=0)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test,y_pred)
    accuracy,precision,recall,accuracyP,accuracyN,balanced_accuracy = bin_cm_params(cm,y_pred)
    classification_result.append(['RandomForestClassifier',balanced_accuracy])
    #LogisticRegression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test,y_pred)
    accuracy,precision,recall,accuracyP,accuracyN,balanced_accuracy = bin_cm_params(cm,y_pred)
    classification_result.append(['LogisticRegression',balanced_accuracy])
    #SVM
    classifier = sklearn.svm.SVC(kernel='linear',random_state=0)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test,y_pred)
    accuracy,precision,recall,accuracyP,accuracyN,balanced_accuracy = bin_cm_params(cm,y_pred)
    classification_result.append(['SVM_linear',balanced_accuracy])
    #SVM rbf
    classifier = sklearn.svm.SVC(kernel='rbf',random_state=0)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test,y_pred)
    accuracy,precision,recall,accuracyP,accuracyN,balanced_accuracy = bin_cm_params(cm,y_pred)
    classification_result.append(['SVM_rbf',balanced_accuracy])
    #SVM poly
    from sklearn.svm import SVC
    classifier = sklearn.svm.SVC(kernel='poly',random_state=0)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test,y_pred)
    accuracy,precision,recall,accuracyP,accuracyN,balanced_accuracy = bin_cm_params(cm,y_pred)
    classification_result.append(['SVM_poly',balanced_accuracy])
    #SVM sigmoid
    classifier = sklearn.svm.SVC(kernel='sigmoid',random_state=0)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test,y_pred)
    accuracy,precision,recall,accuracyP,accuracyN,balanced_accuracy = bin_cm_params(cm,y_pred)
    classification_result.append(['SVM_sigmoid',balanced_accuracy])
    # GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test,y_pred)
    accuracy,precision,recall,accuracyP,accuracyN,balanced_accuracy = bin_cm_params(cm,y_pred)
    classification_result.append(['GaussianNB',balanced_accuracy])

    # ADABoost
    classifier = sklearn.ensemble.AdaBoostClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    accuracy, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred)
    classification_result.append(['ADABoost', balanced_accuracy])


    print('done')
    return classification_result

# KLASYFIKACJA BAZOWA za pomocÄ TRN
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
classifiers_set_result = classifiers_set(X_train,y_train,X_test,y_test)
print('Result for original TRN and TST')

# print(classifiers_set_result)
# print(committees_of_classifiers(classifiers_set_result))

for i in range(len(classifiers_set_result)):
    if i is len(classifiers_set_result) - 1:
        print('===================================')
    print(classifiers_set_result[i])


data = []
bins = []
for i in range(len(classifiers_set_result)):
    data.append(classifiers_set_result[i][1])
    bins.append(classifiers_set_result[i][0])


fig, ax = plt.subplots(figsize=(16, 9))

# Horizontal Bar Plot
ax.barh(bins, data)

# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)

# Remove x, y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)

# Add x, y gridlines
ax.grid(visible=True, color='grey',
        linestyle='-.', linewidth=0.5,
        alpha=0.2)

# Show top values
ax.invert_yaxis()

# Add annotation to bars
for i in ax.patches:
    plt.text(i.get_width(), i.get_y() + 0.5,
             str(round((i.get_width()), 3)),
             fontsize=10, fontweight='bold',
             color='grey')


plt.show()