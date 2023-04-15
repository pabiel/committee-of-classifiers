# scipy lecture notes
# scikit learn
#
# Model: TRN & TST
# Method: Committee of classifiers
#     KNN - canberra
#     naive bias classifier
#     decision tree
#     random forest
#     SVM with RBF
#
# DLA 100 URUCHOMIEŃ RÓZNE WYNIKI RÓZNYCH KLASYFIKATORÓW VS COMEETEEOFCLSF wykres na xlabels 1-100  i linie dla różnych klasyfikacji
# COMEETIEEOFCLFS DLA 1 KLASYFIKATORA DLA 2,3,4,5... do 20 ile tam mam itd.
# https://www.kdnuggets.com/2022/10/ensemble-learning-examples.html

import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.dummy import DummyClassifier


def bin_cm_params(cm, y_pred, y_test):
    tp = cm[0][0]
    fn = cm[0][1]
    fp = cm[1][0]
    tn = cm[1][1]
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1score = (2 * (precision * recall)) / (precision + recall)
    accuracyP = tp / (tp + fn)
    accuracyN = tn / (tn + fp)
    balanced_accuracy = (accuracyP + accuracyN) / 2.
    return round(accuracy, 3), round(f1score, 3), round(precision, 3), round(recall, 3), round(accuracyP, 3), round(accuracyN, 3), round(
        balanced_accuracy, 3)


def average_of_classifiers(classifiers_result):
    result = 0
    no_of_classifiers = 0
    for i in range(len(classifiers_result)):
        result += classifiers_result[i][1]
        no_of_classifiers += 1
    return result/no_of_classifiers


def classifiers_set(X_train, y_train, X_test, y_test):
    classification_result = []
    lst = []
    for i in range(1, 6, 2):
        # kNNClassifier(canberra)
        classifier = KNeighborsClassifier(metric="canberra", n_neighbors=i)
        lst.append(classifier)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cm = metrics.confusion_matrix(y_test, y_pred)
        accuracy, f1score, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred, y_test)
        classification_result.append([str(i)+'KNN_canberra', balanced_accuracy])

        # kNNClassifier(euclidean)
        classifier = KNeighborsClassifier(metric="euclidean", n_neighbors=i)
        lst.append(classifier)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cm = metrics.confusion_matrix(y_test, y_pred)
        accuracy, f1score, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred, y_test)
        classification_result.append([str(i)+'KNN_euclidean', balanced_accuracy])

        # kNNClassifier(manhattan)
        classifier = KNeighborsClassifier(metric="manhattan", n_neighbors=i)
        lst.append(classifier)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cm = metrics.confusion_matrix(y_test, y_pred)
        accuracy, f1score, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred, y_test)
        classification_result.append([str(i)+'KNN_manhattan', balanced_accuracy])

        # kNNClassifier(cityblock)
        classifier = KNeighborsClassifier(metric="cityblock", n_neighbors=i)
        lst.append(classifier)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cm = metrics.confusion_matrix(y_test, y_pred)
        accuracy, f1score, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred, y_test)
        classification_result.append([str(i)+'KNN_cityblock', balanced_accuracy])

        # kNNClassifier(cosine)
        classifier = KNeighborsClassifier(metric="cosine", n_neighbors=i)
        lst.append(classifier)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cm = metrics.confusion_matrix(y_test, y_pred)
        accuracy, f1score, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred, y_test)
        classification_result.append([str(i)+'KNN_cosine', balanced_accuracy])

        # kNNClassifier(l1)
        classifier = KNeighborsClassifier(metric="l1", n_neighbors=i)
        lst.append(classifier)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cm = metrics.confusion_matrix(y_test, y_pred)
        accuracy, f1score, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred, y_test)
        classification_result.append([str(i)+'KNN_l1', balanced_accuracy])

        # kNNClassifier(l2)
        classifier = KNeighborsClassifier(metric="l2", n_neighbors=i)
        lst.append(classifier)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cm = metrics.confusion_matrix(y_test, y_pred)
        accuracy, f1score, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred, y_test)
        classification_result.append([str(i)+'KNN_l2', balanced_accuracy])

        # kNNClassifier(nan_euclidean)
        classifier = KNeighborsClassifier(metric="nan_euclidean", n_neighbors=i)
        lst.append(classifier)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cm = metrics.confusion_matrix(y_test, y_pred)
        accuracy, f1score, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred, y_test)
        classification_result.append([str(i)+'KNN_nan_euclidean', balanced_accuracy])

    # DummyClassifier
    classifier = DummyClassifier(strategy="most_frequent")
    lst.append(classifier)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    accuracy, f1score, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred, y_test)
    classification_result.append(['DummyClassifier', balanced_accuracy])

    # GaussianNB
    classifier = GaussianNB()
    lst.append(classifier)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    accuracy, f1score, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred, y_test)
    classification_result.append(['GaussianNB', balanced_accuracy])

    # DecisionTreeClassifier
    classifier = sklearn.tree.DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=2)
    lst.append(classifier)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    accuracy, f1score, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred, y_test)
    classification_result.append(['DecisionTreeClassifier', balanced_accuracy])

    # RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=10, max_depth=4, criterion='entropy', random_state=0)
    lst.append(classifier)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    accuracy, f1score, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred, y_test)
    classification_result.append(['RandomForestClassifier', balanced_accuracy])


    # LogisticRegression
    classifier = LogisticRegression(random_state=0)
    lst.append(classifier)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    accuracy, f1score, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred, y_test)
    classification_result.append(['LogisticRegression', balanced_accuracy])

    # SVM
    classifier = sklearn.svm.SVC(kernel='linear', random_state=0)
    lst.append(classifier)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)

    accuracy, f1score, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred, y_test)
    classification_result.append(['SVM_linear', balanced_accuracy])

    # SVM rbf
    classifier = sklearn.svm.SVC(kernel='rbf', random_state=0)
    lst.append(classifier)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    accuracy, f1score, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred, y_test)
    classification_result.append(['SVM_rbf', balanced_accuracy])

    # SVM poly
    classifier = sklearn.svm.SVC(kernel='poly', random_state=0)
    lst.append(classifier)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    accuracy, f1score, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred, y_test)
    classification_result.append(['SVM_poly', balanced_accuracy])

    # SVM sigmoid
    classifier = sklearn.svm.SVC(kernel='sigmoid', random_state=0)
    lst.append(classifier)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    accuracy, f1score, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred, y_test)
    classification_result.append(['SVM_sigmoid', balanced_accuracy])

    # ADABoost
    classifier = sklearn.ensemble.AdaBoostClassifier()
    lst.append(classifier)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    accuracy, f1score, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred, y_test)
    classification_result.append(['ADABoost', balanced_accuracy])

    # BaggingClassifier
    classifier = sklearn.ensemble.BaggingClassifier()
    lst.append(classifier)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    accuracy, f1score, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred, y_test)
    classification_result.append(['Bagging', balanced_accuracy])

    estimators = []
    for i in range(len(lst)):
        estimators.append((classification_result[i][0], lst[i]))

    # CommitteeOfClassifiers
    classifier = VotingClassifier(
        estimators=estimators,
        voting="hard")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    accuracy, f1score1, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred, y_test)
    classification_result.append(['CommitteeOfClassifiers', balanced_accuracy])

    return classification_result, lst


def start():
    dataset = pd.read_csv('heart_df.dat', sep=' ', dtype=str)
    # print(dataset)
    attr_no = len(dataset.iloc[1])
    # print(attr_no)
    dec_index = attr_no - 1
    obj_no = len(dataset)

    names = []
    for i in range(1, attr_no): names.append('a' + str(i))
    names.append('class')

    # print(dataset.columns[0])
    # print(dataset.columns)

    classes = np.unique(dataset.to_numpy()[:, attr_no - 1])
    # print(classes)
    class_size_orig = []
    for i in range(0, len(classes)):
        class_size_orig.append(sum(dataset.iloc[:, dec_index] == classes[i]))
    minimal_class_size = min(class_size_orig)

    # print("minimal_class_size = ", minimal_class_size)
    # print("names", names)
    dataset_list = dataset.values.tolist()
    # print(type(dataset_list))
    # print(dataset_list[0])

    # BASIC SPLIT TO TRN AND TST
    train, test = train_test_split(dataset, test_size=0.2) # random_state=10
    train_list = train.values.tolist()
    y_train = train['class']
    y_test = test['class']
    X_train = train.drop(columns=['class'])
    X_test = test.drop(columns=['class'])

    # KLASYFIKACJA BAZOWA za pomoca TRN

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    classifiers_set_result = classifiers_set(X_train, y_train, X_test, y_test)[0]
    print('Result for original TRN and TST')

    for i in range(len(classifiers_set_result)):
        if i is len(classifiers_set_result)-1:
            print('===================================')
        print(classifiers_set_result[i])

    return classifiers_set_result


def add_to_lists(classifiers_set_result):
    for i in range(len(classifiers_set_result)):
        if classifiers_set_result[i][0] == "1KNN_canberra":
            tabKnn.append(classifiers_set_result[i][1])
        elif classifiers_set_result[i][0] == "GaussianNB":
            tabNb.append(classifiers_set_result[i][1])
        elif classifiers_set_result[i][0] == "DecisionTreeClassifier":
            tabDt.append(classifiers_set_result[i][1])
        elif classifiers_set_result[i][0] == "RandomForestClassifier":
            tabRf.append(classifiers_set_result[i][1])
        elif classifiers_set_result[i][0] == "SVM_rbf":
            tabSvm.append(classifiers_set_result[i][1])
        elif classifiers_set_result[i][0] == "ADABoost":
            tabAda.append(classifiers_set_result[i][1])
        elif classifiers_set_result[i][0] == "CommitteeOfClassifiers":
            tabCoc.append(classifiers_set_result[i][1])
        elif classifiers_set_result[i][0] == "Bagging":
            tabBag.append(classifiers_set_result[i][1])
        elif classifiers_set_result[i][0] == "DummyClassifier":
            tabDum.append(classifiers_set_result[i][1])


def coc_num_clfs():
    dataset = pd.read_csv('heart_df.dat', sep=' ', dtype=str)
    train, test = train_test_split(dataset, test_size=0.2,
                                   # random_state=10
                                   )
    train_list = train.values.tolist()
    y_train = train['class']
    y_test = test['class']
    X_train = train.drop(columns=['class'])
    X_test = test.drop(columns=['class'])
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    x = classifiers_set(X_train, y_train, X_test, y_test)[0]
    y = classifiers_set(X_train, y_train, X_test, y_test)[1]
    esti = []
    z = []
    for i in range(len(y)):
        esti.append((x[i][0], y[i]))
        classifier = VotingClassifier(
            estimators=esti,
            voting="hard")
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cm = metrics.confusion_matrix(y_test, y_pred)
        accuracy, f1score1, precision, recall, accuracyP, accuracyN, balanced_accuracy = bin_cm_params(cm, y_pred,
                                                                                                       y_test)
        comm_tab_increase.append(['CommitteeOf ' + str(i + 1) + ' Classifiers', balanced_accuracy])
        z.append(balanced_accuracy)

    fig1, ax1 = plt.subplots()
    k = np.arange(0,len(y),1)
    print(k)
    print(z)
    ax1.scatter(k, z)
    ax1.plot(k, z)
    plt.show()

    return comm_tab_increase


comm_tab_increase = []
tabKnn = []
tabNb = []
tabDt = []
tabRf = []
tabSvm = []
tabCoc = []
tabAda = []
tabBag = []
tabDum = []

for i in range(0,10):
    add_to_lists(start())

a = coc_num_clfs()
print(a)

tabKnn.sort()
tabNb.sort()
tabDt.sort()
tabRf.sort()
tabSvm.sort()
tabCoc.sort()
tabAda.sort()
tabBag.sort()
tabDum.sort()

print(tabKnn)
print(tabNb)
print(tabDt)
print(tabRf)
print(tabSvm)
print(tabCoc)
print(tabAda)
print(tabBag)
print(tabDum)

# for i in range(0,100):
#     start()

t = np.arange(0.0, 10, 1)
knn_model = np.poly1d(np.polyfit(t, tabKnn, 5))
nb_model = np.poly1d(np.polyfit(t, tabNb, 5))
dt_model = np.poly1d(np.polyfit(t, tabDt, 5))
rf_model = np.poly1d(np.polyfit(t, tabRf, 5))
svm_model = np.poly1d(np.polyfit(t, tabSvm, 5))
coc_model = np.poly1d(np.polyfit(t, tabCoc, 5))
ada_model = np.poly1d(np.polyfit(t, tabAda, 5))
bag_model = np.poly1d(np.polyfit(t, tabBag, 5))
dum_model = np.poly1d(np.polyfit(t, tabDum, 5))
myline = np.linspace(0, 9)

fig2, ax2 = plt.subplots()
# ax.plot(t, tabKnn, )
ax2.plot(myline, knn_model(myline), label='KNN')
# plt.scatter(t, tabKnn, label='KNN')

# ax.plot(t, tabNb)
ax2.plot(myline, nb_model(myline), label='NB')
# plt.scatter(t, tabNb, label='NB')

# # ax.plot(t, tabDt)
ax2.plot(myline, dt_model(myline), label='DT')
# plt.scatter(t, tabDt, label='DT')

# # ax.plot(t, tabRf)
ax2.plot(myline, rf_model(myline), label='RF')
# plt.scatter(t, tabRf, label='RF')

# # ax.plot(t, tabSvm)
ax2.plot(myline, svm_model(myline), label='SVM')
# plt.scatter(t, tabSvm, label='SVM')

# # ax.plot(t, tabCoc)
ax2.plot(myline, coc_model(myline), label='COC')
# plt.scatter(t, tabCoc, label='COC')

# # ax.plot(t, tabCoc)
ax2.plot(myline, ada_model(myline), label='ADA')
# plt.scatter(t, tabCoc, label='COC')


# # ax.plot(t, tabCoc)
ax2.plot(myline, bag_model(myline), label='BAG')
# plt.scatter(t, tabCoc, label='COC')


# # ax.plot(t, tabDum)
ax2.plot(myline, dum_model(myline), label='DUM')
# plt.scatter(t, tabDum, label='DUM')


leg = plt.legend()
# ax.set(xlabel='x', ylabel='y',
#        title='100')
# ax.grid()
plt.show()


