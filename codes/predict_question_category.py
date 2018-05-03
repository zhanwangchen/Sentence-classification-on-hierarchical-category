#!/usr/bin/python
# -*- coding: utf-8 -*-


import csv
import sys
import os
import numpy
import numpy as np
import random
from pathlib import Path
import pickle

def getQuestionList(qfile = 'questions.csv'):
  # open questions file
  print("Reading file {} for input.".format(qfile))
  file_question = open(qfile, 'r')
  reader_question = csv.reader(file_question, delimiter=',', quotechar='"', escapechar='\\', doublequote=False)
  next(reader_question)  # first line contains the column names
  n_questions = 0  # number of questions
  questionList = []
  for line in reader_question:  # count number of questions
    n_questions += 1
    questionList.append(line[4])
  file_question.close()
  print("Read Done. Got {} questions for testing.".format(n_questions))
  return questionList

def predict_question_category(qfile = 'questions.csv'):
  questionList = getQuestionList(qfile)
  feature_type = "ngram"
  my_file1 = Path("./data/meanEnsembelClassifier{}.plk".format(feature_type))

  if my_file1.is_file():
    with open(my_file1, 'rb') as f:
      print("Loading the model file {}".format(my_file1))
      minor_eclf, major_eclf, Countvec, sk = pickle.load(f)
      print("Performing the input feature transforming.")
      XTest = Countvec.transform(questionList)
      XTest = sk.transform(XTest)
      print("Predicting begin.")
      parentCatclfLS = major_eclf.predict_proba(XTest)
      parentCatIdPredictLS = major_eclf.predict(XTest)
      subCatclfLS = minor_eclf.predict_proba(XTest)
      subCatIdPredictLS = minor_eclf.predict(XTest)

      predictions = []
      for i in range(subCatIdPredictLS.shape[0]):
        d = {}
        d['minor_category'] = subCatIdPredictLS[i]
        d['confidence_minor_cat'] = subCatclfLS[i].max()

        d['major_category'] = parentCatIdPredictLS[i]
        d['confidence_major_cat'] = parentCatclfLS[i].max()
        predictions.append(d)
      print("predictions returned")
      return predictions
  else:
    print("Model file {} is not exist, please check again, or make sure you are in the correct directory.".format(my_file1))
    exit(-1)




# ***************************** main program *******************************************
# if __name__ == "__main__":
#   print(predict_question_category(qfile = 'question_test.csv'))
