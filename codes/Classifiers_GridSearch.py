import numpy as np
import scipy
import time
from datetime import datetime

import sys

import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import sklearn.svm
import pandas as pd
import pandas
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedShuffleSplit
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


random_state = np.random.RandomState(0)
from sklearn.externals import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
##################################################################

featuresFile='./data/features_fastText.npz'
random_state=0
Test = False
##################################################################

class Classifiers_GridSearch:
    def __init__(self, featuresFile=featuresFile):

        time0 = time.time()
        with np.load(featuresFile) as file:
            self.cat_dict = file['categories']
            self.features = file['features'].T
            self.featurenames = file['featurenames']
            self.cats = file['categoryids'].T.ravel()

            if Test:
                self.features = self.features[:500]
                self.cats =  self.cats[:500]

        print('Loading done', np.around(time.time() - time0, 2), 's')



    def saveModel(self, model, fname):
        time0 = time.time()
        timetag = str(time.strftime("D%d-%H-%M-%S", time.gmtime()))
        fname = timetag + fname
        joblib.dump(model, fname)

    def timeit(method):

        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()

            # print("{} {} {} {}".format(method.__name__, args, kw, te-ts))
            print("{} took time: {} mins".format(method.__name__, np.round((te - ts) / 60, 2)))
            return result

        return timed

    @timeit
    def GridCVSearchNB(self):

        NB_parameters = [
            {'alpha': np.linspace(1.0, 5.0, 10)
             }]

        clf = GridSearchCV(MultinomialNB(),
                           NB_parameters,
                           cv=3,
                           scoring='f1_micro',
                           n_jobs=-1,
                           verbose=2)
        clf.fit(self.features, self.cats)
        self.report(clf, model="GridCVSearchNB")

    @timeit
    def GridCVSearchSVM(self):

        parameters_rbf = [
            {'kernel': ['rbf'],
             'gamma': [0.0005, 0.001, 0.0015],
             'C': np.arange(25, 35, 2.5),
             'class_weight': ['balanced']}]


        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.15, random_state=0)

        clf = sklearn.model_selection.GridSearchCV(SVC(),
                                                   parameters_rbf,
                                                   cv=sss,
                                                   scoring='f1_micro',
                                                   n_jobs=-1,
                                                   verbose=2)
        clf.fit(self.features, self.cats)

        self.report(clf, model="GridCVSearchSVM")

    @timeit
    def GridCVSearchSVMpoly(self):

        parameters_poly = [
            {'kernel': ['poly'], 'degree': [3, 4, 5], 'gamma': [1e-3], 'coef0': [1, 1.05, 1.1, 1.15],
             'C': np.arange(30, 31, 2), 'class_weight': ['balanced']}]


        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.15, random_state=0)
        clf = sklearn.model_selection.GridSearchCV(SVC(),
                                                   parameters_poly,
                                                   cv=sss,
                                                   scoring='f1_micro',
                                                   n_jobs=-1,
                                                   verbose=2)
        clf.fit(self.features, self.cats)

        self.report(clf, model="GridCVSearchSVMpoly")

    @timeit
    def RandomizedGridCVSearchRF(self):


        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.15, random_state=0)

        n_estimators_range = [int(pos) for pos in np.linspace(1, 60, 45)]
        criterion_range = ["gini", "entropy"]
        max_feats_range = ["sqrt", "log2", None]

        # param_grid = dict(clf__n_estimators=n_estimators_range, clf__criterion=criterion_range,
        #                  clf__max_features=max_feats_range)

        param_dist = {"n_estimators": [3, 10, 15, 20, 24, 30, 35],
                      "max_depth": [3, 10, None],
                      "max_features": [int(pos) for pos in np.linspace(1, 300, 45)],
                      "min_samples_split": [int(pos) for pos in np.linspace(2, 50, 45)],
                      "min_samples_leaf": [int(pos) for pos in np.linspace(2, 50, 45)],
                      "bootstrap": [True, False],
                      "criterion": ["gini", "entropy"]}
        n_iter_search = 200
        clf = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_dist,
                                 n_iter=n_iter_search,
                                 cv=sss,
                                 scoring='f1_micro',
                                 n_jobs=-1,
                                 )

        clf.fit(self.features, self.cats)
        self.report(clf, model="RandomizedGridCVSearchRF")

    def fastTextEvalation(self):
        # X,y,Countvec = getBagOfword()
        # y.values.shape

        # plotROC(rfclf,X,y.values, title="RandomForest bag of word ROC")
        from sklearn.model_selection import cross_validate
        from sklearn.metrics import recall_score
        X, y = self.features, self.cats
        # X,y=X[:1500],y[:1500]
        RBFparams = {'kernel': 'rbf', 'C': 27.5, 'gamma': 0.0015, 'class_weight': 'balanced'}
        polyParams = {'kernel': 'poly', 'C': 30, 'degree': 5, 'coef0': 1.15, 'gamma': 0.001, 'class_weight': 'balanced'}
        RBFsvm = svm.SVC(**RBFparams, random_state=random_state)
        polysvm = svm.SVC(**polyParams, random_state=random_state)
        RandomForest = {'bootstrap': False, 'min_samples_leaf': 8, 'n_estimators': 20, 'max_features': 123,
                        'criterion': 'gini', 'min_samples_split': 4, 'max_depth': None}

        rfclf = RandomForestClassifier(**RFParams, random_state=random_state)

        scoring = ['f1_micro', "accuracy"]
        from sklearn.neural_network import MLPClassifier

        MLPclf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                               hidden_layer_sizes=(25, 5), random_state=1)
        from sklearn.naive_bayes import MultinomialNB
        clfNB = MultinomialNB()
        clfs = [RBFsvm, polysvm, rfclf, MLPclf, clfNB]
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=1)
        allScores = {}
        clfNames = ["RBFsvm", "polysvm", "RandomForest", "CNN", "MultNB"]

        for clf, clfName in zip(clfs, clfNames):
            scores = cross_validate(clf, X, y, scoring=scoring,
                                    cv=sss, return_train_score=True)
            allScores[clfName] = scores
        # sorted_scores=sorted(scores.keys())
        d1 = {}
        d2 = {}
        d3 = {}
        d4 = {}
        d5 = {}
        d6 = {}
        d7 = {}
        for clfName in clfNames:
            if clfName != "CNN":
                d1[clfName] = allScores[clfName]["fit_time"]

            d2[clfName] = allScores[clfName]["score_time"]

            d3[clfName] = allScores[clfName]["train_accuracy"]
            d4[clfName] = allScores[clfName]["test_accuracy"]
            d5[clfName] = allScores[clfName]["train_f1_micro"]
            d6[clfName] = allScores[clfName]["test_f1_micro"]
            train = np.mean(allScores[clfName]["train_f1_micro"])
            test = np.mean(allScores[clfName]["test_f1_micro"])
            d7[clfName] = train - test

        # print(allScores)
        # ,d2,d3,d4,d5,d6]
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(11, 3), ncols=2, nrows=1, sharey=True)

        sns.boxplot(data=pd.DataFrame(data=d1), orient="v", ax=ax[0])
        ax[0].set_title('fittime')
        ax[0].set_ylabel('time [Milliseconds]')

        sns.boxplot(data=pd.DataFrame(data=d2), orient="v", ax=ax[1])
        ax[1].set_title('score_time')
        # ax[1].set_ylabel('time')

        fig2, axx = plt.subplots(figsize=(13, 7), ncols=2, nrows=2, sharex=True, sharey=True)

        sns.boxplot(data=pd.DataFrame(data=d3), orient="v", ax=axx[0][0])
        axx[0][0].set_title('train_accuracy')
        axx[0][0].set_ylabel('time [Milliseconds]')

        sns.boxplot(data=pd.DataFrame(data=d4), orient="v", ax=axx[0][1])
        axx[0][1].set_title('test_accuracy')

        sns.boxplot(data=pd.DataFrame(data=d5), orient="v", ax=axx[1][0])
        axx[1][0].set_title('train_f1_micro')

        axx[1][0].set_ylabel('time [Milliseconds]')

        sns.boxplot(data=pd.DataFrame(data=d6), orient="v", ax=axx[1][1])
        axx[1][1].set_title('test_f1_micro')

        fig3, axx1 = plt.subplots(ncols=1, nrows=1, )
        # figsize=(13,7),
        sns.barplot(data=pd.DataFrame(data=d7, index=[0]), ax=axx1)
        axx1.set_title('generalization error')

        plt.tight_layout()
        fig.show()
        fig2.show()
        # ax
        # ax.set_ylabel('time')

    def report(self, clf, model):

        self.saveModel(clf, fname=model)

        print("Grid scores on development set:\n")

        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        print("\nBest parameters set found on development set:\n")

        print(clf.best_score_, clf.best_params_)

    @timeit
    def runAllModel(self):
        # self.GridCVSearchNB()
        self.GridCVSearchSVM()
        self.GridCVSearchSVMpoly()
        self.RandomizedGridCVSearchRF()



if __name__ == "__main__":
    print('Started at', datetime.now().time())
    classifiers_Gs = Classifiers_GridSearch()
    classifiers_Gs.runAllModel()
# eval.LDA()
# eval.PCA()
# eval.KernelPCA()
# exit()

# eval.CV()
# eval.GridCVSearchSVM()



# Evaluation on 10.000 features
# {'kernel': 'rbf', 'class_weight': 'balanced', 'C': 30, 'gamma': 0.001} = 0.606
# 0.572 (+/-0.028) for {'C': 35, 'coef0': 1, 'class_weight': 'balanced', 'degree': 3, 'kernel': 'poly', 'gamma': 0.001}
# 0.606 (+/-0.051) for {'C': 30, 'gamma': 0.001, 'coef0': 1, 'degree': 2, 'kernel': 'poly', 'class_weight': 'balanced'}

# RBF Kernel
# 0.627833180823 {'class_weight': 'balanced', 'kernel': 'rbf', 'C': 53, 'gamma': 0.055}

# Poly Kernel
# 0.608 (+/-0.059) for {'class_weight': 'balanced', 'C': 30, 'gamma': 0.001, 'coef0': 0.8, 'degree': 3, 'kernel': 'poly'}

