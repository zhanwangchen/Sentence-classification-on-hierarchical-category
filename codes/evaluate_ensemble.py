#!/usr/bin/python
# -*- coding: utf-8 -*-
# Test for milestone 3 of the Machine Learning Project.
# Author: Jan Saputra Mueller, Daniel Bartz

import csv
import sys
import os
import numpy
import numpy as np
import math

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


from predict_question_category import *

def pltTimeSaveWithcorrectionDifferenSpeed(thresholdC, predictions,rates, major_labels):
    thresholdC = np.sort(thresholdC) #ascending  np.array(thresholdC)[::-1]
    xSecond = 10
    # expect time used by human for all data
    totalquesiontNum = len(predictions)  # *10000000
    expTimeAll = xSecond * totalquesiontNum
    actualTime = xSecond * rates[:, 1] * totalquesiontNum
    timeSavePercent = 1 - (1.0 * actualTime) / expTimeAll
    # wrong predictoins have to be corrcted later ,this takes y > x seconds.
    ySecondLs = np.linspace(1.01, 2.5, 15) * xSecond
    timeSavePercentList = np.zeros((len(ySecondLs), len(thresholdC) ) )
    timeSavePercentList = []
    for ySecond in ySecondLs:
        #ySecond = xSecond * 2.5
        correctionTime = ySecond * rates[:, 0] * (totalquesiontNum - (rates[:, 1] * totalquesiontNum))
        # correctionTime=0
        timeSavePercent2 = 1 - ((1.0 * actualTime) + correctionTime) / expTimeAll
        timeSavePercent2 = timeSavePercent2[::-1]
        timeSavePercentList.append(timeSavePercent2)


    ###############################
    timeSavePercentList = np.array(timeSavePercentList)

    print(timeSavePercentList.shape)
    print(timeSavePercentList)
    import seaborn as sns
    t= ySecondLs-xSecond #vmin=0, vmax=1,
    thresholdCLabel=[]
    for v in thresholdC:
        thresholdCLabel.append("{:2.1f}".format(v))
    TLabel = []
    for v in t:
        TLabel.append("{:2.2f}".format(v))

    ax = sns.heatmap(timeSavePercentList,  xticklabels=thresholdCLabel, yticklabels=TLabel, annot=True)
    ax.set(xlabel='acceptable error rate', ylabel='Time t=y-x, (dirrerent speeds)')
    plt.title("Time saving (percentage) human&machine, dirrerent speeds ")


    plt.show()
    fig = plt.figure()
    fig.savefig("/tmp/TimeSavingPercentHumanMachineDirrerentSpeeds.pdf")
    plt.clf()
    #############
    fig, ax = plt.subplots()
    cax = ax.imshow(timeSavePercentList,  cmap=plt.cm.hot)
    plt.sca(ax)
    #plt.yticks(range(len(t)), t)
    #plt.xticks(range(len(thresholdC)), thresholdC)
    #plt.xticks(rotation=45)
    #ax.tick_params(axis='x', **kwargs)
    plt.title("Time saving (percentage) human&machine, dirrerent speeds ")
    fig.tight_layout()

    for x in range(timeSavePercentList.shape[0]):
        for y in range(timeSavePercentList.shape[1]):
            c = "black"
            if timeSavePercentList[x, y] < 0.2:
                c="white"
            plt.text(x, y , '%.2f' % timeSavePercentList[x, y],
                     horizontalalignment='center',
                     verticalalignment='center',
                     color=c
                     )
    cbar = fig.colorbar(cax)
    plt.show()
    fig.savefig("/tmp/TimeSavingDiffSpeed.pdf")
    plt.clf()
    #########

def pltTimeSave(thresholdC, rates, predictions, major_labels):

    xSecond = 10
    #expect time used by human for all data
    totalquesiontNum = len(predictions) #*10000000
    expTimeAll = xSecond * totalquesiontNum
    actualTime = xSecond * rates[:, 1]*totalquesiontNum
    timeSavePercent = 1 - (1.0 * actualTime)/expTimeAll

    ###############################
    ySecond = xSecond * 80
    correctionTime = ySecond * rates[:, 0] * (totalquesiontNum - (rates[:, 1] * totalquesiontNum))
    #correctionTime=0
    timeSavePercent2 = 1 - ((1.0 * actualTime) + correctionTime) / expTimeAll
    timeSavePercent2=timeSavePercent2[::-1]
    yps= []
    for i ,var in enumerate(thresholdC):
        if var in [0.05, 0.1, 0.2]:
            yps.append(timeSavePercent2[i])


    plt.plot(thresholdC, timeSavePercent2, label="with human correction")
    #plt.plot([0.05, 0.1, 0.2], yps, 'ro')
    ####################################################
    # #wrong predictoins have to be corrcted later ,this takes y > x seconds.
    # ySecond = xSecond *2
    # timeSavePercentList=[]
    #
    # thresholdC = np.sort(np.array(list(thresholdC)))  # ascending
    # for confidence_level in thresholdC:
    #     major = [x['confidence_major_cat'] > confidence_level for x in predictions]
    #
    #     major_predictions_filtered = filter_mask(predictions, major)
    #
    #     #the prediction that under the certain thresholde, the come to human, take x seconds.
    #     humanFirstPrecitionTime  = (len(predictions) - len(major_predictions_filtered)) * xSecond
    #     predictionIds = [x['major_category']  for x in major_predictions_filtered]
    #     major_true_filtered = filter_mask(major_labels, major)
    #     wrongPredictionNum = 0
    #     if len(major_true_filtered)!=0:
    #         wrongPredictionNum  = len(predictions)- accuracy_score(major_true_filtered, predictionIds, normalize=False)
    #     HumanCorrectionTime = wrongPredictionNum * ySecond
    #
    #     # expect time used by human for all data
    #     expTimeAll = xSecond * len(predictions)
    #     timeSavePercentWC = 1 - 1.0 * (humanFirstPrecitionTime + HumanCorrectionTime) / expTimeAll
    #     timeSavePercentList.append(timeSavePercentWC)
    # yps= []
    # for i ,var in enumerate(thresholdC):
    #     if var in [0.05, 0.1, 0.2]:
    #         yps.append(timeSavePercentList[i])
    #
    #
    # plt.plot(thresholdC, timeSavePercentList, label="with human correction")
    # plt.plot([0.05, 0.1, 0.2], yps, 'ro')

    ####################################################
    timeSavePercent=timeSavePercent[::-1]
    yps2 = []
    for i, var in enumerate(thresholdC):
        if var in [0.05, 0.1, 0.2]:
            yps2.append(timeSavePercent[i])
    plt.plot(thresholdC, timeSavePercent, label="without further correction", linestyle=":")
    #plt.plot([0.05, 0.1, 0.2], yps2, 'bo')

    plt.xlabel("acceptable error rate")
    plt.ylabel("percentage of time saving")
    plt.title(" Time saving (percentage) ")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show()
    fig = plt.figure()
    fig.savefig("TimeSavingPercent.pdf")
    plt.clf()

def pltTimeSaveWithcorrection(thresholdC, predictions, major_labels):
    xSecond = 10

    #wrong predictoins have to be corrcted later ,this takes y > x seconds.
    ySecond = 20
    timeSavePercentList=[]
    thresholdC = np.sort(thresholdC)  # ascending
    for confidence_level in thresholdC:
        major = [x['confidence_major_cat'] > confidence_level for x in predictions]

        major_predictions_filtered = filter_mask(predictions, major)

        #the prediction that under the certain thresholde, the come to human, take x seconds.
        humanFirstPrecitionTime  = (len(predictions) - len(major_predictions_filtered)) * xSecond
        predictionIds = [x['major_category']  for x in major_predictions_filtered]
        major_true_filtered = filter_mask(major_labels, major)
        wrongPredictionNum = 0
        if len(major_true_filtered)!=0:
            wrongPredictionNum  = len(predictions)- accuracy_score(major_true_filtered, predictionIds, normalize=False)
        HumanCorrectionTime = wrongPredictionNum * ySecond

        # expect time used by human for all data
        expTimeAll = xSecond * len(predictions)
        timeSavePercent = 1 - 1.0 * (humanFirstPrecitionTime + HumanCorrectionTime) / expTimeAll
        timeSavePercentList.append(timeSavePercent)

    plt.plot(thresholdC, timeSavePercentList)

    plt.xlabel("acceptable error rate")
    plt.ylabel("percentage of time saving")
    plt.title(" Time saving (percentage), with human correction")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show()
    fig = plt.figure()
    fig.savefig("TimeSavingPercentWithCorrection.pdf")
    plt.clf()

# def pltTimeSaveWithcorrectionDifferenSpeed(thresholdC, predictions, major_labels):
#     thresholdC = np.sort(thresholdC) #ascending  np.array(thresholdC)[::-1]
#     xSecond = 10
#
#     # wrong predictoins have to be corrcted later ,this takes y > x seconds.
#     ySecondLs = np.linspace(1.1, 4,10) * xSecond
#     timeSavePercentList = np.zeros((len(thresholdC), len(ySecondLs)) )
#     for j, ySecond in enumerate(ySecondLs):
#         for i, confidence_level in enumerate(thresholdC):
#             major = [x['confidence_major_cat'] > confidence_level for x in predictions]
#
#             major_predictions_filtered = filter_mask(predictions, major)
#
#             # the prediction that under the certain thresholde, the come to human, take x seconds.
#             humanFirstPrecitionTime = (len(predictions) - len(major_predictions_filtered)) * xSecond
#
#             major_true_filtered = filter_mask(major_labels, major)
#             predictionIds = [x['major_category'] for x in major_predictions_filtered]
#             wrongPredictionNum = 0
#             if len(major_true_filtered) != 0:
#                 wrongPredictionNum = len(predictions) - accuracy_score(major_true_filtered, predictionIds,
#                                                                    normalize=False)
#             HumanCorrectionTime = wrongPredictionNum * ySecond
#
#             # expect time used by human for all data
#             expTimeAll = xSecond * len(predictions)
#             timeSavePercent = 1 - 1.0 * (humanFirstPrecitionTime + HumanCorrectionTime) / expTimeAll
#             timeSavePercentList[i,j]=timeSavePercent
#
#     print(timeSavePercentList)
#     import seaborn as sns
#     t= ySecondLs-xSecond #vmin=0, vmax=1,
#     ax = sns.heatmap(timeSavePercentList,  xticklabels=thresholdC, yticklabels=t)
#     ax.set(xlabel='acceptable error rate', ylabel='Time t=y-x, (dirrerent speeds)')
#     plt.title("Time saving (percentage) human&machine, dirrerent speeds ")
#
#
#     plt.show()
#     fig = plt.figure()
#     fig.savefig("/tmp/TimeSavingPercentHumanMachineDirrerentSpeeds.pdf")
#     plt.clf()

def pltClassifiersCorrelations(qfile):
    names =   ["MNB", "RBFsvm", "Polysvm", "RandomForest" ]


    questionList=getQuestionList(qfile)
    #subCatIdPredictLS, subCatclfLS, parentCatIdPredictLS, parentCatclfLS=getVotingClassifierPrecitions(questionList)
    FeatureTypeLs=["TFIDF", "bagOfWord", "ngram"]
    #FeatureTypeLs = ["ngram"]
    predictionLSAllFeatures = []
    namesLs=[]
    for FeatureType in FeatureTypeLs:

        file = Path("./data/{}predictionsLs".format(FeatureType))
        genNew = False
        if  file.is_file() or not genNew:
            print("loading existed file {}".format(file))
            # [subCatclfLS, parentCatclfLS] = np.load(file)
            subCatclfLS = []
            parentCatclfLS = []
            with open(file, 'rb') as f:
                [Names, predictionLs, clfLs] = pickle.load(f)
                predictionLSAllFeatures += predictionLs
                namesLs+=Names
                continue


        print("using feature type {}".format(FeatureType))
        [X, parentCatY, subCatY, XTest] = getfeatures(questionList, name=FeatureType)
        estimatorLs = [('MNB', MNB()), ('RBFsvm', RBFsvm()), ('Polysvm', Polysvm()),
                       ('RandomForest', myRandomForest())]
        ensemClf = VotingClassifier(
            estimators=estimatorLs,
            voting='soft')

        cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

        #for Ytrain, MajoyMinorCat in zip([parentCatY, subCatY], ["major", "minor"]):
        for Ytrain, MajoyMinorCat in zip([parentCatY], ["major"]):

            for train_index, test_index in cv.split(X):
                AllclfScores = []
                Names = []
                predictionLs = []
                clfLs=[]
                for (name, clf) in estimatorLs:
                    print("TRAIN:", len(train_index), "TEST:", len(test_index))
                    clf = clf.fit(X[train_index], Ytrain[train_index])
                    clfLs.append(clf)
                    predictions = clf.predict(X[test_index])
                    predictionLs.append(predictions)
                    predictionLSAllFeatures.append(predictions)
                    # score = cross_val_score(clf, X, Ytrain, cv=cv, scoring='f1_micro')
                    # AllclfScores.append(np.mean(score))
                    Names.append(name)
                    namesLs += Names
                    # print("{} {} classifier f1 score micor {}".format(MajoyMinorCat, name, np.mean(score)))
                file="./data/{}predictionsLs".format(FeatureType)
                with open(file, 'wb') as f:
                    pickle.dump([Names, predictionLs,clfLs], f)
                    print("saved " + file)
                    exit(-1)

    corr_coeff = np.corrcoef(predictionLSAllFeatures)
    namesLs=[]
    for i in ["TFIDF", "bOW", "ngram"]:
        for j in ["MNB", "RBFsvm", "Polysvm", "RForest" ]:
            namesLs.append(i+j)
    fig, ax = plt.subplots()
    cax = ax.imshow(corr_coeff, interpolation='nearest', cmap=plt.cm.hot)
    plt.sca(ax)
    plt.yticks(range(len(namesLs)), namesLs)
    plt.xticks(range(len(namesLs)), namesLs)
    plt.xticks(rotation=45)
    #ax.tick_params(axis='x', **kwargs)
    plt.title("Correlations for different Classifiers")
    fig.tight_layout()

    for x in range(corr_coeff.shape[0]):
        for y in range(corr_coeff.shape[1]):
            c = "black"
            if corr_coeff[x, y] < 0.2:
                c="white"
            plt.text(x, y , '%.2f' % corr_coeff[x, y],
                     horizontalalignment='center',
                     verticalalignment='center',
                     color=c
                     )
    cbar = fig.colorbar(cax)
    plt.show()
    fig.savefig("/tmp/{}corrmat.pdf".format("all"))
    plt.clf()
    # score = cross_val_score(ensemClf, X, Ytrain, cv=cv, scoring='f1_micro')
    Names.append("averageEnsemble")
    # AllclfScores.append(np.mean(score))
    # print("{} ensemble classifier f1 micor {}".format(MajoyMinorCat, np.mean(score)))

#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     sns.set()
#
#     # Load the example flights dataset and conver to long-form
#
#
#     # Draw a heatmap with the numeric values in each cell
#     f, ax = plt.subplots(figsize=(9, 6))
#     g=sns.heatmap(corr_coeff, annot=True, linewidths=.5, ax=ax)
#     plt.yticks(range(len(namesLs)), namesLs)
#     g.set_yticklabels(namesLs, rotation=60)
#
#     plt.xticks(range(len(namesLs)), namesLs)
#     #plt.yticks(rotation=90)
# #(rotation=30)
#     plt.xticks(rotation=45)
#     #ax.tick_params(axis='x', **kwargs)
#     plt.title("Correlations for different Classifiers")
#     plt.show()






# report: evaluation 2, confidence vary accross catag
def pltCategConfidence(predictions, cateIdToName_major, cateIdToName_minor):
    major = [x['confidence_major_cat'] for x in predictions]
    majorCateNames = []
    minorCateNames = []
    for x in predictions:

            majorCateNames.append(  cateIdToName_major.get(x['major_category'], "-1")  )

            minorCateNames.append( cateIdToName_minor.get(x['minor_category'], "-1" ) )


    d = {'major_confidence': major, 'majorCateNames': majorCateNames}

    df = pd.DataFrame(data=d)#by='major_confidence',
    df.groupby([majorCateNames])['major_confidence'].mean().sort_values(ascending=False).plot(kind='bar',title="major_category_confidence")
    plt.ylabel("confidence")
    plt.xticks(rotation=45)
    plt.show()
    newNamels = []
    for name in minorCateNames:
        newNamels.append(name.split(" ")[0])
    minorCateNames = newNamels
    d = {'minor_confidence': major, 'minorCateNames': minorCateNames}

    df = pd.DataFrame(data=d)#by='minor_confidence',
    df.groupby([minorCateNames])['minor_confidence'].mean().sort_values( ascending=False)[:15].plot(
        kind='bar', title="minor_category_confidence")
    plt.ylabel("confidence")
    plt.xticks(rotation=40)
    plt.show()

def pltconfusion_matrix(predictions,cateIdToName_major, cateIdToName_minor, major_labels, minor_labels):
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix

    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt

    major = [x['confidence_major_cat'] for x in predictions]
    majorCateNames = []
    minorCateNames = []
    y_pred2=[]
    major_labels2=[]
    for i, x in enumerate(predictions):
            if x['major_category']=="-1" or x['major_category']== -1 or major_labels[i]==-1 or major_labels[i]=="-1":
                continue
            majorCateNames.append(  cateIdToName_major.get(x['major_category'], "-1")  )

            minorCateNames.append( cateIdToName_minor.get(x['minor_category'], "-1" ) )
            y_pred2.append( cateIdToName_major.get(x['major_category'], "-1") )
            major_labels2.append(cateIdToName_major.get(major_labels[i], "-1"))
    print(majorCateNames)
    print(classification_report(y_pred2, major_labels2))

    y_true = major_labels
    y_pred = [x['major_category'] for x in predictions]

    #print(classification_report(y_true, y_pred, target_names=set(majorCateNames)))

    cmat = confusion_matrix(y_true, y_pred)
    cmat = confusion_matrix(y_pred2, major_labels2)
    print(cmat)

    df_cm = pd.DataFrame(cmat)
    #plt.figure(figsize=(14, 14))
    #g = sn.heatmap(df_cm, annot=True)
    majorCateNames=[]
    keySet = set(cateIdToName_major.keys())
    keySet.remove(-1)
    keys = list(keySet)
    keys.sort()
    SortedNames = []
    for i in keys:
        SortedNames.append(cateIdToName_major.get(i,"-1"))

    print(SortedNames)
    print(len(SortedNames))
    #sns.set(font_scale=0.8)

    ###################
    # fig, ax = plt.subplots()
    # cmat=np.array(cmat)
    # #cmap=plt.cm.hot
    # cax = ax.imshow(cmat ,cmap=plt.cm.hot )
    # plt.sca(ax)
    # #plt.yticks(range(len(t)), t)
    # #plt.xticks(range(len(thresholdC)), thresholdC)
    # #plt.xticks(rotation=45)
    # #ax.tick_params(axis='x', **kwargs)
    #
    #
    # for x in range(cmat.shape[0]):
    #     for y in range(cmat.shape[1]):
    #         c = "black"
    #         if cmat[x, y] < 0.2:
    #             c="white"
    #         plt.text(x, y , '%3.f' % cmat[x, y],
    #                  horizontalalignment='center',
    #                  verticalalignment='center',
    #                  color=c
    #                  )
    # cbar = fig.colorbar(cax)
    # plt.show()
    # exit(-1)
    #############

    g=sn.heatmap(cmat, annot=True, annot_kws={"size": 7}, fmt='g')
    #g.setxx  xticklabels=SortedNames, yticklabels=SortedNames
    #fig, ax = plt.subplots()

    #cax = ax.imshow(cmat, interpolation='nearest', cmap=plt.cm.hot)
    #plt.sca(ax)

    tick_marks = np.arange(len(SortedNames))
    plt.title("confusion matrix on major catagory")
    plt.xticks(tick_marks, SortedNames, rotation=45)
    plt.yticks(tick_marks[::-1], SortedNames, rotation=1)
    plt.show()

    # , index = [i for i in "ABCDEFGHIJK"],
    # columns = [i for i in "ABCDEFGHIJK"]

    # d = {'major_confidence': major, 'majorCateNames': majorCateNames}
    #
    # df = pd.DataFrame(data=d)#by='major_confidence',
    # df.groupby([majorCateNames])['major_confidence'].mean().sort_values(ascending=False).plot(kind='bar',title="major_category_confidence")
    # plt.ylabel("confidence")
    # plt.xticks(rotation=45)
    # plt.show()
    # newNamels = []
    # for name in minorCateNames:
    #     newNamels.append(name.split(" ")[0])
    # minorCateNames = newNamels
    # d = {'minor_confidence': major, 'minorCateNames': minorCateNames}
    #
    # df = pd.DataFrame(data=d)#by='minor_confidence',
    # df.groupby([minorCateNames])['minor_confidence'].mean().sort_values( ascending=False)[:15].plot(
    #     kind='bar', title="minor_category_confidence")
    # plt.ylabel("confidence")
    # plt.xticks(rotation=40)
    # plt.show()


def getCataIDToNameDic(major_categories, minor_categories, major_labels, minor_labels):
    major_cateIdToName = dict()
    minor_cateIdToName = dict()
    # dict2 = {'Sex': 'female'}
    #
    # dict.update(dict2)
    for i in np.arange(len(major_labels)):
        dictTmp = {major_labels[i]: major_categories.get(major_labels[i],"-1") }
        major_cateIdToName.update(dictTmp)
        dictTmp2 = {minor_labels[i]: minor_categories.get(minor_labels[i],"-1") }
        minor_cateIdToName.update(dictTmp2)
    return major_cateIdToName, minor_cateIdToName



def milestone3_test(qfile = 'question_train_subsample.csv', qcatfile = 'question_category_train_subsample.csv', catfile = 'category.csv'):
  ''' Test for milestone 3. This test uses your classifier to predict categories for a test data set,
      and does some evaluation.

     input
       qfile     file with test questions
       qcatfile  file with categories for the test questions (needed for evaluation)
       catfile   file with categories
  '''
  #
  # ClassifierComparision
  # questionList = getQuestionList(qfile)
  # ClassifierComparision(questionList)
  #


  # read table category
  file_category = open(catfile, 'r')
  reader_category = csv.reader(file_category, delimiter=',', quotechar='"', escapechar='\\', doublequote=False)
  major_categories = {} # major categories dictionary: category id -> major category name
  minor_categories = {} # minor categories dictionary: category id -> minor category name
  minor_major = {} # dictionary: minor category -> major category
  next(reader_category) # skip first line
  for line in reader_category:
    if line[1] == '0':
      # category is a major category
      major_categories[int(line[0])] = line[2]
    else:
      # category is a minor category
      minor_categories[int(line[0])] = line[2]
      minor_major[int(line[0])] = int(line[1])
  file_category.close()

  # get true "question -> minor category" labeling
  file_qc = open(qcatfile, 'r')
  reader_qc = csv.reader(file_qc, delimiter=',', quotechar='"', escapechar='\\', doublequote=False)
  qc = {} # map: question id -> category id
  question_category_columns = next(reader_qc) # first line contains the column names
  for line in reader_qc:
    qc[int(line[2])] = int(line[1])
  file_qc.close()

  # read questions
  file_question = open(qfile, 'r')
  reader_question = csv.reader(file_question, delimiter=',', quotechar='"', escapechar='\\', doublequote=False)
  major_labels = [] # true major labels
  minor_labels = [] # true minor labels
  next(reader_question) # first line contains the column names
  for line in reader_question:
    qid = int(line[0])
    if qid in qc:
      label = qc[qid]
      minor_labels.append(label)
      major_labels.append(minor_major[label])
    else:
      # question has no category
      minor_labels.append(-1)
      major_labels.append(-1)
  file_question.close()

  # make predictions
  predictions = predict_question_category(qfile)

  # print confidences
  thresholdC = np.linspace(0.0, 1.0, num= 10)
  thresholdC = thresholdC.tolist()
  thresholdC = set(thresholdC)
  # thresholdC.add(0.2)
  # thresholdC.add(0.1)
  # thresholdC.add(0.05)
  thresholdC = np.sort(np.array(list(thresholdC)))  # ascending

  #thresholdC.append(1.0)
  rates=[]
  for c in thresholdC:
    [major_error_rate, major_rejection_rate, minor_error_rate, minor_rejection_rate]= score_with_confidence(predictions, c, major_categories, minor_categories, major_labels, minor_labels)
    rates.append([major_error_rate, major_rejection_rate, minor_error_rate, minor_rejection_rate])

  y = np.array(rates)

  #report : prediction methodology
  # How correlated are the errors of the different classifiers?
  #pltClassifiersCorrelations(qfile)
  #exit(-1)

  #report : evaluation 2, confidence vary across categories,
  major_cateIdToName, minor_cateIdToName=getCataIDToNameDic(major_categories, minor_categories, major_labels, minor_labels)
  #pltCategConfidence(predictions,major_cateIdToName, minor_cateIdToName)

  #report : plot confusion matrix,
  #major_cateIdToName, minor_cateIdToName=getCataIDToNameDic(major_categories, minor_categories, major_labels, minor_labels)
  #pltconfusion_matrix(predictions,major_cateIdToName, minor_cateIdToName, major_labels, minor_labels)
  #exit(-1)

  #report : evaluation 3, time saving x,
  pltTimeSave(thresholdC, y, predictions,major_labels)

  # report : evaluation 4, time saving y
  #pltTimeSaveWithcorrection(thresholdC, predictions, major_labels)

  # with different speeds
  #pltTimeSaveWithcorrectionDifferenSpeed(thresholdC, predictions, y,major_labels)

  plt.plot(thresholdC, y[:, 0], label='major_error_rate', linestyle=':', color='green' )
  plt.plot(thresholdC, y[:, 1], label='major_rejection_rate')
  plt.plot(thresholdC, y[:, 2], label='minor_error_rate', linestyle='--')
  plt.plot(thresholdC, y[:, 3], label='minor_rejection_rate', linestyle='-.', color='red')
  plt.xlabel("threshold")
  plt.ylabel("test error and rejection rete")
  plt.title(" Test error and rejection rates")
  plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

  plt.show()
  fig = plt.figure()
  fig.savefig("test_error_and_rejection_rete.pdf")
  plt.clf()


def score_with_confidence(predictions, confidence_level, major_categories, minor_categories, major_labels, minor_labels):
  ''' Prints out some information about the prediction confidence '''

  sys.stdout.write('==== Confidence level: ' + percent1(confidence_level) + '\n')
  # get questions with confidence > confidence_level
  major = [x['confidence_major_cat'] > confidence_level for x in predictions]
  minor = [x['confidence_minor_cat'] > confidence_level for x in predictions]

  sys.stdout.write('Number of classified data points (major classes): ' + str(positives(major)) + '/' + str(len(predictions)) + '\n')
  sys.stdout.write('Number of classified data points (minor classes): ' + str(positives(minor)) + '/' + str(len(predictions)) + '\n')

  major_rejection_rate= 1 - 1.0 * positives(major) /len(predictions)
  minor_rejection_rate= 1 - 1.0 * positives(minor) /len(predictions)


  major_predictions_filtered = filter_mask(predictions, major)
  minor_predictions_filtered = filter_mask(predictions, minor)


  major_truelabels_filtered = filter_mask(major_labels, major)
  minor_truelabels_filtered = filter_mask(minor_labels, minor)

  major_error_rate = 1 - f1_score(major_truelabels_filtered, [x['major_category'] for x in major_predictions_filtered], average='micro')
  minor_error_rate = 1 - f1_score(minor_truelabels_filtered, [x['minor_category'] for x in minor_predictions_filtered], average='micro')



  (mean, std) = mean_std([x['confidence_major_cat'] for x in major_predictions_filtered])
  sys.stdout.write('Confidence major classes: mean=' + percent1(mean) + ' std=' + percent1(std) + '\n')
  (mean, std) = mean_std([x['confidence_minor_cat'] for x in minor_predictions_filtered])
  sys.stdout.write('Confidence minor classes: mean=' + percent1(mean) + ' std=' + percent1(std) + '\n')
  return [major_error_rate, major_rejection_rate, minor_error_rate, minor_rejection_rate]


  # # major categories
  # sys.stdout.write('\n== Errors on major classes\n')
  # major_predictions_ids = [x['major_category'] for x in major_predictions_filtered]
  # for category in major_categories:
  #   sys.stdout.write(major_categories[category]
  #                       + align_names(len(major_categories[category]), 30)
  #                       + ': FPR=' + percent1(fpr(major_predictions_ids, major_labels_filtered, category))
  #                       + ' TPR=' + percent1(tpr(major_predictions_ids, major_labels_filtered, category)) + '\n')
  #
  # # minor categories
  # sys.stdout.write('\n== Errors on minor classes\n')
  # minor_predictions_ids = [x['minor_category'] for x in minor_predictions_filtered]
  # for category in minor_categories:
  #   sys.stdout.write(minor_categories[category]
  #                       + align_names(len(minor_categories[category]), 30)
  #                       + ': FPR=' + percent1(fpr(minor_predictions_ids, minor_labels_filtered, category))
  #                       + ' TPR=' + percent1(tpr(minor_predictions_ids, minor_labels_filtered, category)) + '\n')
  #
  # sys.stdout.write('\n')
  # # ipdb.set_trace()
  # sys.stdout.write('accuracy major: '
  #                   + percent1(accu(  major_predictions_ids, major_labels_filtered )) + '\n')
  # sys.stdout.write('accuracy minor: '
  #                   + percent1(accu(  minor_predictions_ids, minor_labels_filtered )) + '\n')
  # # ipdb.set_trace()
  # sys.stdout.write('\n')

def mean_std(l):
  ''' Returns the mean and std for the elements in l '''
  mean = numpy.mean(list(l))  #reduce(lambda x, y: x + y, l) / float(len(l))
  std = numpy.std(list(l))  #math.sqrt(reduce(lambda x, y: x + y, map(lambda x: (x - mean) ** 2, l)) / (float(len(l)) - 1.0))
  return(mean, std)

def positives(l):
  ''' Counts how often "true" appears in the list l '''
  return numpy.sum(l)  #reduce(lambda x, y: x + y, map(lambda x: 1 if x else 0, l))

def filter_mask(l, m):
  ''' Filters the elements of l according to the mask m (list containing booleans) '''
  # ipdb.set_trace()
  return [l[i] for i in range(len(l)) if m[i]]  #map(lambda x: x[1], filter(lambda x: x[0], zip(m, l)))

def tpr(predictions, true_labels, category):
  ''' Calculates the true positive rate for a category '''
  category_mask = [x == category for x in true_labels]
  true_positives = [x == category for x in filter_mask(predictions, category_mask)]
  n_positive = positives(category_mask)
  if n_positive > 0:
    return float(positives(true_positives)) / float(n_positive)
  else:
    return None

def fpr(predictions, true_labels, category):
  ''' Calculates the false positive rate for a category '''
  not_category_mask = [x != category for x in true_labels]
  false_positives = [x == category for x in filter_mask(predictions, not_category_mask)]
  n_negatives = positives(not_category_mask)
  if n_negatives > 0:
    return float(positives(false_positives)) / float(n_negatives)
  else:
    return None

def percent1(f):
  if f is None:
    return '  None'
  else:
    s = ('%.1f' % (100*f)) + '%'
    s = align_names(len(s), 6) + s
    return s

def accu(predictions, true_labels):
    ''' returns the ratio of correct predictions '''
    return numpy.mean([predictions[i] == true_labels[i] for i in range(len(true_labels))])
    #l = zip(true_labels, predictions)
    #return reduce(lambda x, y: x + y,
    #                 map(lambda x: 1 if x[0] == x[1] else 0, l) )/float(len(l))

def blanks(n):
  s = ''
  for i in range(n):
    s += ' '
  return s

def align_names(n, max_l):
  return blanks(max_l - n)

# ***************************** main program *******************************************
if __name__ == "__main__":

  milestone3_test()

  #pltClassifiersCorrelations(qfile = 'question_train_subsample.csv')

