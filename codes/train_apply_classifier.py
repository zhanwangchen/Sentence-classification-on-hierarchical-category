# coding: utf-8

from extract_features import *
#from classifiers import *
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

from extract_features import getCleanedCategoryData
from extract_features import *

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
random_state = np.random.RandomState(0)
import time;

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        #print("{} {} {} {}".format(method.__name__, args, kw, te-ts))
        print("{} took time: {} mins".format(method.__name__, np.round((te-ts)/60,2) ))
        return result

    return timed

def plotROC(clf,X,y, title="ROC"):


    y = label_binarize(y.tolist(), classes=list(set(y)))
    n_classes = y.shape[1]


    n_samples, n_features = X.shape

    
    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=1)

    for train_index, test_index in sss.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(clf)
    y_score = None
    if isinstance(clf,RandomForestClassifier) or isinstance(clf,GaussianNB):
        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    else:
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    lw = 2

    ##############################################################################
    # Plot of a ROC curve for a specific class
#     plt.figure()

#     plt.plot(fpr[2], tpr[2], color='darkorange',
#              lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
#     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic example')
#     plt.legend(loc="lower right")
#     plt.show()


    ##############################################################################
    # Plot ROC curves for the multiclass problem

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    f, axarr = plt.subplots(1, 2,sharex=True, sharey=True,figsize=(10,5))
    #axarr[0, 0].plot(x, y)
    #axarr[0].set_title('Axis [0,0]')
    axarr[0].plot(fpr["micro"], tpr["micro"],
             label='micro-average(area:{0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    axarr[0].plot(fpr["macro"], tpr["macro"],
             label='macro-average(area:{0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    axarr[0].legend(loc="lower right")
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    #colors=axarr[1]._get_lines.color_cycle
    from cycler import cycler
    #colors=cycler(color='bgrcmyk')
    for i, color in zip(range(n_classes), colors):
        axarr[1].plot(fpr[i], tpr[i], lw=lw,
                 label='class{0} (area:{1:0.2f})'
                 ''.format(i, roc_auc[i]))
        # Shrink current axis by 20%
    box = axarr[1].get_position()
    axarr[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    axarr[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    axarr[0].plot([0, 1], [0, 1], 'k--', lw=lw)
    axarr[1].plot([0, 1], [0, 1], 'k--', lw=lw)
    
    axarr[0].set_xlim([0.0, 1.0])
    axarr[0].set_ylim([0.0, 1.05])
    axarr[1].set_xlim([0.0, 1.0])
    axarr[1].set_ylim([0.0, 1.05])
    
    axarr[0].set_xlabel('False Positive Rate')
    axarr[0].set_ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.suptitle(title,y=1.05, fontsize=17)
    #plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

@timeit   
def RBFsvm(X,y):
    RBFparams = {'kernel': 'rbf', 'C': 27.5, 'gamma': 0.0015, 'class_weight': 'balanced', 'probability': True}
    RBFsvm=svm.SVC(**RBFparams,random_state=random_state)
    RBFsvm.fit(X,y)
    return RBFsvm

@timeit
def Polysvm(X,y):
    polyParams = {'kernel': 'poly', 'C': 30, 'degree': 5, 
                  'coef0': 1.15, 'gamma': 0.001, 'class_weight': 'balanced', 'probability': True}
    polysvm=svm.SVC(**polyParams,random_state=random_state)
    polysvm.fit(X,y)
    return polysvm

@timeit
def myRandomForest(X,y):
    RFParams = {'bootstrap': False, 'min_samples_leaf': 8, 
                'n_estimators': 20, 'max_features': 123, 
                'criterion': 'gini', 'min_samples_split': 4, 'max_depth': None}

    clf = RandomForestClassifier(**RFParams,random_state=random_state)
    clf.fit(X,y)
    return clf

@timeit
def MNB(X,y):
    clf = MultinomialNB(alpha=4.44)
    clf.fit(X,y)
    return clf

@timeit
def GNB(X,y):
    clf = GaussianNB()
    clf.fit(X,y)
    return clf

###########################classifiers without fitted
def RBFsvm():
    RBFparams = {'kernel': 'rbf', 'C': 27.5, 'gamma': 0.0015, 'class_weight': 'balanced', 'probability': True}
    RBFsvm = svm.SVC(**RBFparams, random_state=random_state)

    return RBFsvm


def Polysvm():
    polyParams = {'kernel': 'poly', 'C': 30, 'degree': 5,
                  'coef0': 1.15, 'gamma': 0.001, 'class_weight': 'balanced', 'probability': True}
    polysvm = svm.SVC(**polyParams, random_state=random_state)

    return polysvm


def myRandomForest():
    RFParams = {'bootstrap': False, 'min_samples_leaf': 8,
                'n_estimators': 20, 'max_features': 123,
                'criterion': 'gini', 'min_samples_split': 4, 'max_depth': None}

    clf = RandomForestClassifier(**RFParams, random_state=random_state)

    return clf


def MNB():
    clf = MultinomialNB(alpha=4.44)

    return clf


def GNB():
    clf = GaussianNB()
    return clf
# @timeit
# def CNN(X,y):
#     from CNN_keras import getDataAndTrain
#     clf = CNN_keras.getDataAndTrain()
#     return clf

###########################classifiers without fitted

def trainClassifier(clf,X,y):
    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=1)
    for train_index, test_index in sss.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
    return clf




#features=bagOfWord| FastText| TFIDF

def train_apply_classifier(classifier = 'RBFsvm', qfile_train='question_train.csv',
                           qcatfile_train='question_category_train.csv',
                           catfile='category.csv',qfile_test = 'question_test.csv', 
                           subcats=False, features="bagOfWord"
                           ):
    X,y,Xtest = getfeatures(name=features)
    print("using classifier {}".format(classifier))
    predition=None
    if(classifier == 'RBFsvm'):
        clf = RBFsvm(X,y)
        predition = clf.predict(Xtest)

    elif(classifier == 'Polysvm'):
        clf = Polysvm(X,y)
        predition = clf.predict(Xtest)

    elif(classifier == 'MultinomialNB' and features!="FastText" and features!="word2vec"):
        clf = MNB(X,y)
        predition = clf.predict(Xtest)

    elif(classifier == 'GaussianNB'):
        from scipy.sparse import csr_matrix
        if isinstance(X,csr_matrix):
            X=X.toarray()
            Xtest = Xtest.toarray()
        clf = GNB(X,y)
        predition = clf.predict(Xtest)

    elif(classifier == 'RandomForest'):
        clf = myRandomForest(X,y)
        predition = clf.predict(Xtest)

    elif(classifier == 'CNN'):
        clf = CNN(X,y)
        predition = clf.predict(Xtest)
    else:
        print("There are wrong parameters.")


    print(predition)
    np.save("./data/preditionRes",predition)
    return predition
    
    
if __name__ == "__main__":
    train_apply_classifier(classifier = 'RBFsvm', qfile_train='question_train.csv',
                           qcatfile_train='question_category_train.csv',
                           catfile='category.csv',qfile_test = 'question_test.csv', 
                           subcats=False, features="FastText"
                           )   
#getfeatures("bagOfWord")
#for classifier in ['RBFsvm','Polysvm','MultinomialNB','GaussianNB','RandomForest']:
#if __name__ == "__main__":
#    for classifier in ['RBFsvm','Polysvm','MultinomialNB','GaussianNB','RandomForest']:
#
#        train_apply_classifier(classifier = classifier, qfile_train='question_train.csv',
#                               qcatfile_train='question_category_train.csv',
#                               catfile='category.csv',qfile_test = 'question_test.csv', 
#                               subcats=False, features="bagOfWord"
#                               )