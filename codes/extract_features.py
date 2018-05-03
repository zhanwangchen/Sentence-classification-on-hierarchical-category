# coding: utf-8
# -*- coding: utf-8 -*-
import pandas
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import numpy
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from nltk.stem import SnowballStemmer
porter = SnowballStemmer("german")
import enchant
import warnings
warnings.filterwarnings("ignore")

#fastTextFile = "./models/wiki.de.vec.test"
#fastTextFile = "./models/wiki.de.vec"
fastTextFile = "./models/wiki.de.bin"
word2vecFile = "./models/wiki.de.vec"

def getCleanedCategoryData(fname="category.csv"):
    f =""
    import re
    out = open("category_cleaned.csv","w")
    count=1
    for line in open(fname):

        if count==1:
            f += line.strip() + "\n"
            count+=1
            continue
        if count!=1:
            if(re.search(r'\d+-\d+-\d+ \d+:\d+:\d+"$', line.strip())):
                f += line.strip() +"\n"
                
            else:
                f += line.strip()
      
    out.write(f)
    out.close()

    import csv
    with open('category_cleaned.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
       
        df1 = []
        for row in reader:
            df1.append([row["category_id"],row["parent_id"], 
                        row["category_name"],row["headline"] ])

        
    df1  = pd.DataFrame(df1,columns=["category_id", "parent_id","category_name","headline"])
    df1[["category_id", "parent_id"]] = df1[["category_id", "parent_id"]].apply(pd.to_numeric, errors='coerce')

    category = df1
    return category


import string
import re

def cleanupDoc(s):
    
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u":\)"  # flags (iOS)
        u":\-\)"  # flags (iOS)
        u"\.\:\-\)"  # flags (iOS)
        u"\?"
        u"\\\\"
        u"\"\""
                               "]+")
    s = emoji_pattern.sub(r' ', s)
    stopset = set(stopwords.words('german'))
    german_tokenizer = nltk.data.load(resource_url='tokenizers/punkt/german.pickle')
    tokens = german_tokenizer.tokenize(s)
    tokens =  nltk.tokenize.WordPunctTokenizer().tokenize(s)
    punctuations = list(string.punctuation)
    
    cleanup = [token.lower() for token in tokens if token.lower() not in stopset and  len(token)>1]
    cleanup = [i for i in cleanup if i not in punctuations]
    cleanup = " ".join(cleanup)
    
      
    cleanup = re.sub("\d", "",cleanup)
    return cleanup

enchantD = enchant.Dict("de_DE")
def correctWord(sentence):
    
    sents = []
    for word in sentence.split(" "): 
        
        if len(word.strip()) == 0:
            continue
            
        if not enchantD.check(word):
           
            if(len(enchantD.suggest(word))):
                sents.append(enchantD.suggest(word)[0])
                
            else:
                sents.append(word)
               
        else:
            sents.append(word)
           
    return ' '.join(sents) 


from nltk.stem import SnowballStemmer
porter = SnowballStemmer("german")

def WordStem(sentence):
    StemSentence = []
    for word in sentence.split(" "):
        StemSentence.append(porter.stem(word).lower())
    return ' '.join(StemSentence)

def TranslateCh(sentence):
    dt = {ord('ä') : 'ae', ord('ü'): 'ue', ord('ö'):'oe', ord('ß'): 'ss'}
   
    return sentence.translate(dt)

def getCleanedQuestion_trainData(fname="question_train.csv",
                                 Stem=False, 
                                 correct=False,
                                 trans=True):
    
    question_train = pd.read_csv(fname,quotechar='"', error_bad_lines=False)
    #question_train = question_train[0:10]
    question_train = question_train[["question","category_main_id","categories"]]

    question_train["category_main_id"] = question_train["category_main_id"].apply(pd.to_numeric,errors="coerce")
    question_train = question_train.dropna()
    question_train["category_main_id"] = question_train['category_main_id'].astype('int')
    
    
   
    question_train['question'] = question_train['question'].astype('str')
    newQuestion = []
    for i, sent in question_train['question'].iteritems():
     
        sent = cleanupDoc(sent)
        if correct:
            sent = correctWord(sent)
            if i%500==0:
                print("correctWord row Nr."+str(i))
    
      
        if Stem:
            sent = WordStem(sent)
            if i%500==0:
                print("WordSteming row Nr."+str(i))
        if trans:
            sent =  TranslateCh(sent)
        
        newQuestion.append(sent)
    question_train['question'] = newQuestion
    return question_train
 



def mergeQuestion_category(question_train, category):
    merged = pd.merge(question_train,category,
                      left_on="category_main_id",
                      right_on="category_id",
                      how="inner")

    return merged





def feature_extraction(X_train):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,)

    X_train = vectorizer.fit_transform(X_train)
    return X_train, vectorizer.get_feature_names()
    





def feature_selection(merged, subcats=False, m="ch2", percentile=30,save=True,fname="features.npz"):
    
    from sklearn.feature_selection import mutual_info_classif, chi2
    from sklearn.feature_selection import SelectKBest, SelectPercentile
    X_train = merged['question']
    
    if subcats:
        Y_train=merged["category_main_id"]
    else:
        Y_train=merged["parent_id"]
        
    TfidfFeatures, feature_names = feature_extraction(X_train)
    
    vectorizedFeatures = TfidfFeatures
    if m=="ch2":
        m = chi2
    else:
        m = mutual_info_classif(discrete_features=True)
        
    if percentile!=None:
        sk = SelectPercentile(m, percentile=percentile)
        X_trainSelected = sk.fit_transform(vectorizedFeatures, Y_train)
        
    else:
       
    
        sk = SelectKBest(m, k=5000)
        X_trainSelected = sk.fit_transform(vectorizedFeatures, Y_train)
    supportIdx = sk.get_support(indices=True)
    #print(type(feature_names))
    feature_namesSelected = [feature_names[i] for i in supportIdx]
    if save:
        
        features = X_trainSelected.toarray().T
        featurenames = feature_namesSelected
        if subcats:
            categoryids = merged['category_main_id'].values.reshape(1,merged['category_main_id'].values.shape[0])
            categories = dict(zip(merged['category_main_id'].values,merged['categories'].values))
        else:
            categoryids = merged['parent_id'].values.reshape(1,merged['parent_id'].values.shape[0])
            categories = dict(zip(merged['parent_id'].values,merged['categories'].values))
        np.savez(fname, features=features, 
                 featurenames=featurenames, categoryids=categoryids, categories=categories)
                #mutual_info_classif(X_train, Y_train, discrete_features=True)
    print(X_trainSelected.shape)
    return X_trainSelected, feature_namesSelected


def feature_selection2(X, y, XTest, m="ch2", percentile=20, fname="./data/features_trainTest_TFIDF.npz"):
    from sklearn.feature_selection import mutual_info_classif, chi2
    from sklearn.feature_selection import SelectKBest, SelectPercentile



    if m == "ch2":
        m = chi2
    else:
        m = mutual_info_classif(discrete_features=True)
    sk = None
    if percentile != None:
        sk = SelectPercentile(m, percentile=percentile)
        X = sk.fit_transform(X, y)

    else:

        sk = SelectKBest(m, k=5000)
        X = sk.fit_transform(X, y)
    # supportIdx = sk.get_support(indices=True)
    # # print(type(feature_names))
    # feature_namesSelected = [feature_names[i] for i in supportIdx]
    print("after feature select {}%, xtrain shape is {}".format(percentile,X.shape))
    return X, sk.transform(XTest)



def classifierReport(X_train, Y_train,):
    from sklearn import metrics
    from sklearn.model_selection import cross_val_score

    from sklearn import svm
    from sklearn import metrics
    from sklearn.cross_validation import train_test_split
    from sklearn.datasets import load_iris
    from sklearn.metrics import f1_score, accuracy_score
    
    gnb = GaussianNB()
    mnb = MultinomialNB()
    svc = svm.SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma=0.01, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
    svc = svm.SVC()
    clf = mnb
    clfs=[]
    #clfs.append(svc)
    clfs.append(gnb)
    #clfs.append(gnb)
    
    #clfs.append(svc)
    #scoring='f1'
    #if type(X_train) is numpy
    
        
    for clf in clfs:
        if isinstance(X_train, np.ndarray):
            scores = cross_val_score(clf, X_train, Y_train, cv=3 )
        else:
            scores = cross_val_score(clf, X_train.toarray(), Y_train, cv=3 )
        print('Accuracy:', np.mean(scores))
    return scores


def extract_features(qfile='question_train.csv',
    qcatfile='question_category_train.csv',
    catfile='category.csv',
    subcats=True,
    outfile='features.npz'):
    
    category=getCleanedCategoryData()
    question_train = getCleanedQuestion_trainData()  
    
    merged=mergeQuestion_category(question_train, category)
    
    TfidfFeatures, feature_names = feature_extraction(merged['question'])
    
    X_trainSelected, feature_namesSelected = feature_selection(merged=merged,save=True,fname=outfile)
    print("extract_features done!")
    return
    if subcats:
        classifierReport(X_trainSelected, merged['category_main_id'])
    else:
        classifierReport(X_trainSelected, merged['parent_id'])


def fastTestsupervised():
    from fastText import train_supervised

    f = open("fasttestTrain.txt",'w')
    #for i = range(len(merged)):
    #    merged[][] 
    for _, row in merged[["category_main_id","question"]].iterrows():
        f.write("__label__"+str(int(row['category_main_id']))+" "+row['question']+"\n")
    f.close()

    #model_unsup.save_model(<path>)

    model_sup = train_supervised(
        input="fasttestTrain.txt",
        epoch=1,
        thread=10
    )
    s = "wann wurde rudolf hess geboren früher bedeutung"
    v= model_sup.get_sentence_vector()
    print(v)
    print(model_sup.predict(s))
    return model_sup

def word2vecModel(fname=word2vecFile):
    import gensim.models.word2vec
    import time;
    s = time.time()
    print("loading:  {}".format(fname))
    #import word2vec
    #new_model = gIn [2]: model = gensim.models.Word2Vec.load_word2vec_format('/data5/momo-projects/user_interest_classification/code/word2vec/vectors_groups_1105.bin', binary=True, unicode_errors='ignore')
    #m = gensim.models.Word2Vec.load(fname)#*
    #m = gensim.models.word2vec.load_word2vec_format.load(fname, binary=True, unicode_errors='ignore')
    m = gensim.models.KeyedVectors.load_word2vec_format(fname, unicode_errors='ignore')
    e = time.time()
    print("loaded: took {}".format(e-s))
    return m

def getFastTextModel(fname=fastTextFile):
    from fastText import load_model
    import time;
    s = time.time()
    print("loading:  {}".format(fname))
    m = load_model(fname)
    e = time.time()
    print("loaded: took {}".format(e-s))
    return m

def sentenceToVec(sents, model,m="word2vec", s="avg"):
    if m=="doc2vec":
        tokens_list=sents.split(" ")
        v=model.infer_vector(doc_words=tokens_list, steps=20, alpha=0.025)
        return v

    sentvec = []
    for word in sents.split(" "):
        try:
            if m=="fasttext":
                v = model.get_word_vector(word)
            else:
                v = model[word]
            
        except Exception as inst:
            print(inst)
            #sentvec.append([0])
            continue
            
        sentvec.append(v)
    if s=="avg":
        return np.average(sentvec, axis=0)
    else:
        return np.sum(sentvec, axis=0)

def getEmbedingVec(xTrain, model, m="word2vec", saveVecTo=None):
    sentsvecs =[]
    for row in xTrain:
        sentsvecs.append(sentenceToVec(row ,model,m) )
        
    res = np.zeros((len(sentsvecs),len(sentsvecs[0])))
    for i, r in enumerate(sentsvecs):
        res[i] = sentsvecs[i]
        
    res=pd.DataFrame(res).fillna(0).values         
    if saveVecTo:
        np.savez(saveVecTo, sentenceToVecs=res)
    return res

def TrainDoc2vec(X, Y, saveTo="Doc2vecModelGemsim.bin"):
    #X,Y=merged["question"].values,merged["parent_id"].values
    import gensim
    LabeledSentence = gensim.models.doc2vec.LabeledSentence
    class LabeledLineSentence(object):
        def __init__(self, xTrain,YTrain):
            self.xTrain = xTrain
            self.YTrain = YTrain
        def __iter__(self):
            for i in range(len(self.YTrain)):
                yield LabeledSentence(str(self.xTrain[i]).split(" "), [self.YTrain[i]])
    it = LabeledLineSentence(X, Y)
    model = gensim.models.Doc2Vec(window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025) # use fixed learning rate
    model.build_vocab(it)
    for epoch in range(1):
        model.train(it,total_examples=model.corpus_count,epochs=model.iter)
        model.alpha -= 0.002 # decrease the learning rate
        model.min_alpha = model.alpha # fix the learning rate, no deca
        #model.train(it)
    if saveTo!=None:
        model.save(saveTo)
    return model

def getDoc2vecModel(fname="Doc2vecModelGemsim.bin"):
    model = gensim.models.Doc2Vec
    model.load(fname)
    return model 

def benchmarkAll(subcats=False):
    category=getCleanedCategoryData()
    question_train = getCleanedQuestion_trainData()  
    
    merged=mergeQuestion_category(question_train, category)
# ###########1
    TfidfFeatures, feature_names = feature_extraction(merged['question'])
    subcats =True
    for per in np.linspace(31, 100, num=10,dtype=int):
        if per<31.01:
            continue
        X_trainSelected, feature_namesSelected = feature_selection(
            merged=merged,save=False,fname="",percentile=per)
    
        if subcats:
            Y=merged['category_main_id']
        else:
            Y=merged['parent_id']
        print("{}% percentile".format(per))
        classifierReport(X_trainSelected, Y)
#     return
###########1
###########2
    model = gensim.models.KeyedVectors.load_word2vec_format('../submissionMS1/word2vecGerman.model', 
                                                           binary=True, 
                                                           unicode_errors='ignore')
    sentvecs = getEmbedingVec(merged["question"],model=model)
    classifierReport(sentvecs,Y)
###########2
###########3
    model = gensim.models.KeyedVectors.load_word2vec_format(fastTextFile, unicode_errors='ignore')
    sentvecs = getEmbedingVec(merged["question"],model=model)

    classifierReport(sentvecs,Y)
###########3
###########4

    category=getCleanedCategoryData()
    question_train = getCleanedQuestion_trainData(Stem=False, 
                                 correct=False,
                                 trans=False)  
    
    merged=mergeQuestion_category(question_train, category)
    subcats = True
    if subcats:
        Y=merged['category_main_id']
    else:
        Y=merged['parent_id']
    
#     mFastText = getFastTextModel(fname="../submissionMS1/wiki.de.bin")
#     sentvecs = getEmbedingVec(merged["question"],model=mFastText, m="fasttext")
#     classifierReport(sentvecs,Y)

###########4
###########5
    Doc2vecM=TrainDoc2vec(merged["question"],merged["parent_id"])

    sentvecs=getEmbedingVec(xTrain=merged["question"],m="doc2vec",model=Doc2vecM)
    classifierReport(sentvecs,Y)

###########5
def getBagOfword(qfile='question_train.csv',
    qcatfile='question_category_train.csv',
    catfile='category.csv',
    subcats=False  ):
    
    category=getCleanedCategoryData()
    question_train = getCleanedQuestion_trainData()  
    
    merged=mergeQuestion_category(question_train, category)
    if subcats:
        Y = merged['category_main_id'].values
    else:
        Y = merged['parent_id'].values

    from sklearn.feature_extraction.text import CountVectorizer
    Countvec = CountVectorizer()
    X = Countvec.fit_transform(merged["question"])
    return X,Y,Countvec

def getFastTextFeatures(fileName="./data/features_fastText.npz"):
    with np.load(fileName) as file:
        cat_dict = file['categories'] 
        features = file['features'].T
        featurenames = file['featurenames']
        cats = file['categoryids'].T.ravel()
        X,y=features,cats
        return X,y
    

from pathlib import Path
def getfeatures(name="FastText"):
    print("getting {} features".format(name))
    if name=="FastText":
        X,y = getFastTextFeatures(fileName="./data/features_fastText.npz")
        my_file = Path("./data/TestSetfeatures_fastText.npz")
        if my_file.is_file():
            print("loading existed file {}".format(my_file))
            data = np.load(my_file)
            XTest = data["sentenceToVecs"]
            return X, y, XTest

        XTest = getTestData()
        fastTextM = getFastTextModel()
        XTest=getEmbedingVec(XTest,fastTextM,saveVecTo="./data/TestSetfeatures_fastText.npz",m="fasttext")
        #sentenceToVec(XTest,)
        return X,y,XTest
    
    if name=="word2vec":
        X,y = getFastTextFeatures(fileName="./data/features_fastText.npz")
        my_file1 = Path("./data/TestSetfeatures_word2vec.npz")
        my_file2 = Path("./data/TrainSetfeatures_word2vec.npz")
        if my_file1.is_file() and my_file2.is_file():
            print("loading existed file {} {}".format(my_file1,my_file2))
            data = np.load(my_file1)
            XTest = data["sentenceToVecs"]
            data = np.load(my_file2)
            X = data["sentenceToVecs"]
            y= np.load("./data/parent_id.npy")
            return X, y, XTest
        XTest = getTestData()
        word2vm = word2vecModel()
        getEmbedingVec(XTest,word2vm,saveVecTo="./data/TestSetfeatures_word2vec.npz")
        category=getCleanedCategoryData()
        question_train = getCleanedQuestion_trainData()  

        merged=mergeQuestion_category(question_train, category)

        y = merged['parent_id'].values
        np.save("./data/parent_id",y)

        train=merged["question"].values
        getEmbedingVec(train,word2vm,saveVecTo="./data/TrainSetfeatures_word2vec.npz")
        #sentenceToVec(XTest,)
        return X,y,XTest
    
    if name=="bagOfWord":
        percentile = 60
        file1 = Path("./data/TrainTest_bagOfWord_{}p.npy".format(percentile))
        if file1.is_file():
            print("loading existed file {}".format(file1))
            [X, y, XTest] = np.load(file1)
            return X, y, XTest
        category=getCleanedCategoryData()
        question_train = getCleanedQuestion_trainData()  

        merged=mergeQuestion_category(question_train, category)

        y = merged['parent_id'].values

        from sklearn.feature_extraction.text import CountVectorizer
        train=merged["question"].values
        Countvec = CountVectorizer()
        t=getTestData(fname="question_test.csv")
        test = t.values
        comb = np.concatenate((train, test), axis=0)
        Countvec.fit(comb)
        X=Countvec.transform(train)
        XTest=Countvec.transform(t)
        X, XTest = feature_selection2(X, y, XTest, percentile=percentile)
        np.save("./data/TrainTest_bagOfWord_{}p".format(percentile), [X, y, XTest])
        return X,y,XTest

    if name=="TFIDF":
        percentile=35
        file1 = Path("./data/TrainTest_tfidf_{}p.npy".format(percentile))
        if file1.is_file():
            print("loading existed file {}".format(file1))
            [X, y, XTest] = np.load(file1)
            return X,y,XTest
        category=getCleanedCategoryData()
        question_train = getCleanedQuestion_trainData()

        merged=mergeQuestion_category(question_train, category)

        y = merged['parent_id'].values

        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(sublinear_tf=True, )

        train=merged["question"].values

        t=getTestData(fname="question_test.csv")
        test = t.values
        comb = np.concatenate((train, test), axis=0)
        vectorizer.fit(comb)
        X=vectorizer.transform(train)
        XTest=vectorizer.transform(t)
        X,XTest= feature_selection2(X,y,XTest,percentile=percentile)
        np.save("./data/TrainTest_tfidf_{}p".format(percentile), [X,y,XTest])
        return X,y,XTest

    
def getTestData(fname="question_test.csv"):
    question_test = pd.read_csv(fname, quotechar='"', error_bad_lines=False)
    #question_train = question_train[0:10]
    question_test = question_test[["question"]]

    question_test = question_test.dropna()
 
    
   
    question_test['question'] = question_test['question'].astype('str')
    #print(question_test)
    newQuestion = []
    for i, sent in question_test['question'].iteritems():
     
        sent = cleanupDoc(sent)
        newQuestion.append(sent)
    question_test['question'] = newQuestion
    return question_test['question']

# if __name__ == "__main__":
#     extract_features(qfile='question_train.csv',
#        qcatfile='question_category_train.csv',
#        catfile='category.csv',
#        subcats=False,
#        outfile='features.npz')



#getTestData()
#benchmarkAll()
