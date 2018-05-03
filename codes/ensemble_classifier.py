

from train_apply_classifier import *
from pathlib import Path
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score


import pickle
def getfeatures(questionList, name="FastText"):
    print("getting {} features".format(name))

    genNew = False
    if name == "bagOfWord":

        percentile = 81
        file1 = Path("./data/TrainTest_bagOfWord_{}p.npy".format(percentile))

        if not file1.is_file() or genNew:
            category = getCleanedCategoryData()
            question_train = getCleanedQuestion_trainData()

            merged = mergeQuestion_category(question_train, category)

            subCatY = merged['category_id'].values
            parentCatY = merged['parent_id'].values

            from sklearn.feature_extraction.text import CountVectorizer
            train = merged["question"].values
            Countvec = CountVectorizer()
            # t = getTestData(fname=qfile)
            # test = t.values

            comb = np.concatenate((train, questionList), axis=0)
            Countvec.fit(comb)
            X = Countvec.transform(train)
            XTest = Countvec.transform(questionList)
            X, XTest = feature_selection2(X, subCatY, XTest, percentile=percentile)
            np.save("./data/TrainTest_bagOfWord_{}p".format(percentile), [X, parentCatY, subCatY, XTest])

        print("loading existed file {}".format(file1))
        [X, parentCatY, subCatY, XTest] = np.load(file1)
        return [X, parentCatY, subCatY, XTest]

    if name=="TFIDF":
        percentile=35
        file1 = Path("./data/TrainTest_tfidf_{}p.npy".format(percentile))

        if not file1.is_file() or genNew:
            category = getCleanedCategoryData()
            question_train = getCleanedQuestion_trainData()

            merged = mergeQuestion_category(question_train, category)

            subCatY = merged['category_id'].values
            parentCatY = merged['parent_id'].values

            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(sublinear_tf=True )

            train = merged["question"].values

            # t = getTestData(fname="question_test.csv")
            # test = t.values
            comb = np.concatenate((train, questionList), axis=0)
            vectorizer.fit(comb)
            X = vectorizer.transform(train)
            XTest = vectorizer.transform(questionList)
            X, XTest = feature_selection2(X, subCatY, XTest, percentile=percentile)
            np.save("./data/TrainTest_tfidf_{}p".format(percentile), [X, parentCatY, subCatY, XTest])

        print("loading existed file {}".format(file1))
        [X, parentCatY, subCatY, XTest] = np.load(file1)
        return [X, parentCatY, subCatY, XTest]
        #return X,y,XTest

    if name == "ngram":

        percentile = 81
        file1 = Path("./data/TrainTest_ngram_{}p.npy".format(percentile))

        if not file1.is_file() or genNew:
            category = getCleanedCategoryData()
            question_train = getCleanedQuestion_trainData()

            merged = mergeQuestion_category(question_train, category)

            subCatY = merged['category_id'].values
            parentCatY = merged['parent_id'].values

            from sklearn.feature_extraction.text import CountVectorizer
            train = merged["question"].values
            Countvec = CountVectorizer(ngram_range=(1, 3))
            # t = getTestData(fname=qfile)
            # test = t.values

            comb = np.concatenate((train, questionList), axis=0)
            Countvec.fit(comb)
            X = Countvec.transform(train)
            XTest = Countvec.transform(questionList)
            #X, XTest = feature_selection2(X, subCatY, XTest, percentile=percentile)
            from sklearn.feature_selection import mutual_info_classif, chi2

            sk = SelectKBest(chi2, k=5000)
            X = sk.fit_transform(X, subCatY)
            XTest = sk.transform(XTest)
            np.save("./data/TrainTest_ngram_{}p".format(percentile), [X, parentCatY, subCatY, XTest,Countvec,sk])

        print("loading existed file {}".format(file1))
        [X, parentCatY, subCatY, XTest, Countvec,sk] = np.load(file1)
        return [X, parentCatY, subCatY, XTest, Countvec,sk]

    if name == "FastText":
        X, y = getFastTextFeatures(fileName="./data/features_fastText.npz")
        my_file = Path("./data/TestSetfeatures_fastText.npz")
        if my_file.is_file():
            print("loading existed file {}".format(my_file))
            data = np.load(my_file)
            XTest = data["sentenceToVecs"]
            return X, y, XTest

        XTest = getTestData()
        fastTextM = getFastTextModel()
        XTest = getEmbedingVec(XTest, fastTextM, saveVecTo="./data/TestSetfeatures_fastText.npz", m="fasttext")
        # sentenceToVec(XTest,)
        return X, y, XTest

    if name == "word2vec":
        X, y = getFastTextFeatures(fileName="./data/features_fastText.npz")
        my_file1 = Path("./data/TestSetfeatures_word2vec.npz")
        my_file2 = Path("./data/TrainSetfeatures_word2vec.npz")
        if my_file1.is_file() and my_file2.is_file():
            print("loading existed file {} {}".format(my_file1, my_file2))
            data = np.load(my_file1)
            XTest = data["sentenceToVecs"]
            data = np.load(my_file2)
            X = data["sentenceToVecs"]
            y = np.load("./data/parent_id.npy")
            return X, y, XTest
        XTest = getTestData()
        word2vm = word2vecModel()
        getEmbedingVec(XTest, word2vm, saveVecTo="./data/TestSetfeatures_word2vec.npz")
        category = getCleanedCategoryData()
        question_train = getCleanedQuestion_trainData()

        merged = mergeQuestion_category(question_train, category)

        y = merged['parent_id'].values
        np.save("./data/parent_id", y)

        train = merged["question"].values
        getEmbedingVec(train, word2vm, saveVecTo="./data/TrainSetfeatures_word2vec.npz")
        # sentenceToVec(XTest,)
        return X, y, XTest

# @timeit
# def ClassifierComparision(questionList):
#     #FeatureTypeLs=["TFIDF", "bagOfWord", "ngram"]
#     FeatureTypeLs = ["bagOfWord"]
#     for FeatureType in FeatureTypeLs:
#         print("using feature type {}".format(FeatureType))
#         [X, parentCatY, subCatY, XTest] = getfeatures(questionList, name=FeatureType)
#         estimatorLs = [('MNB', MNB()), ('RBFsvm', RBFsvm()), ('Polysvm', Polysvm()), ('RandomForest', myRandomForest())]
#         ensemClf = VotingClassifier(
#             estimators=estimatorLs,
#             voting='soft')
#
#         cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
#
#         for Ytrain, MajoyMinorCat in zip([parentCatY, subCatY], ["major", "minor"]):
#
#             for train_index, test_index in cv.split(X):
#                 AllclfScores = []
#                 Names = []
#                 predictionLs = []
#                 for (name,clf) in estimatorLs:
#                         print("TRAIN:", train_index, "TEST:", test_index)
#                         clf=clf.fit(X[train_index],Ytrain[train_index])
#                         predictions = clf.predict(X[test_index])
#                         predictionLs.append(predictions)
#                         # score = cross_val_score(clf, X, Ytrain, cv=cv, scoring='f1_micro')
#                         # AllclfScores.append(np.mean(score))
#                         Names.append(name)
#                         # print("{} {} classifier f1 score micor {}".format(MajoyMinorCat, name, np.mean(score)))
#                 # score = cross_val_score(ensemClf, X, Ytrain, cv=cv, scoring='f1_micro')
#                 Names.append("averageEnsemble")
#                 # AllclfScores.append(np.mean(score))
#                 # print("{} ensemble classifier f1 micor {}".format(MajoyMinorCat, np.mean(score)))

@timeit
def ClassifierComparision(questionList):
    #FeatureTypeLs=["TFIDF", "bagOfWord", "ngram"]
    FeatureTypeLs = ["ngram"]
    for FeatureType in FeatureTypeLs:
        print("using feature type {}".format(FeatureType))
        [X, parentCatY, subCatY, XTest, Countvec] = getfeatures(questionList, name=FeatureType)
        estimatorLs = [('MNB', MNB()), ('RBFsvm', RBFsvm()), ('Polysvm', Polysvm()), ('RamdomForest', myRandomForest())]
        ensemClf = VotingClassifier(
            estimators=estimatorLs,
            voting='soft')

        cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
        for Ytrain, MajoyMinorCat in zip([parentCatY, subCatY], ["major", "minor"]):
            AllclfScores=[]
            Names = []
            for (name,clf) in estimatorLs:
                score = cross_val_score(clf, X, Ytrain, cv=cv, scoring='f1_micro')
                AllclfScores.append(np.mean(score))
                Names.append(name)
                print("{} {} classifier f1 score micor {}".format(MajoyMinorCat, name, np.mean(score)))
            score = cross_val_score(ensemClf, X, Ytrain, cv=cv, scoring='f1_micro')
            Names.append("averageEnsemble")
            AllclfScores.append(np.mean(score))
            print("{} ensemble classifier f1 micor {}".format(MajoyMinorCat, np.mean(score)))

            y_pos = np.arange(len(Names))
            #, alpha=0.5 , align='center'
            plt.bar(y_pos, AllclfScores)
            plt.xticks(y_pos, Names)
            plt.xticks(rotation=45)
            plt.ylabel('f1 score micor')
            plt.title('Classifiers F1 score Comparision, {}'.format(MajoyMinorCat))

            plt.show()




def getVotingClassifierPrecitions(questionList):

    feature_type = "ngram"
    my_file1 = Path("./data/meanEnsembelClassifier{}.plk".format(feature_type))

    if my_file1.is_file():
        with open(my_file1, 'rb') as f:
            print("loading existed file {}".format(my_file1))
            minor_eclf, major_eclf, Countvec, sk = pickle.load(f)
            XTest = Countvec.transform(questionList)
            XTest = sk.transform(XTest)
            parentCatclfLS = major_eclf.predict_proba(XTest)
            parentCatIdPredictLS = major_eclf.predict(XTest)
            subCatclfLS = minor_eclf.predict_proba(XTest)
            subCatIdPredictLS = minor_eclf.predict(XTest)
            return subCatIdPredictLS, subCatclfLS, parentCatIdPredictLS, parentCatclfLS

    [X, parentCatY, subCatY, XTest,Countvec,sk] = getfeatures(questionList, name=feature_type)
    # X=X[:1000]
    # parentCatY = parentCatY[:1000]
    # subCatY= subCatY[:1000]
    # XTest= XTest[:1000]
    major_eclf = VotingClassifier(
        estimators=[('MNB', MNB()), ('RBFsvm', RBFsvm()), ('Polysvm', Polysvm()), ('RandomForest', myRandomForest())],
        voting='soft')
    major_eclf = major_eclf.fit(X, parentCatY)
    parentCatclfLS = major_eclf.predict_proba(XTest)
    parentCatIdPredictLS = major_eclf.predict(XTest)

    minor_eclf = VotingClassifier(
        estimators=[('MNB', MNB()), ('RBFsvm', RBFsvm()), ('Polysvm', Polysvm()), ('RandomForest', myRandomForest())],
        voting='soft')
    minor_eclf = minor_eclf.fit(X, subCatY)
    subCatclfLS = minor_eclf.predict_proba(XTest)
    subCatIdPredictLS = minor_eclf.predict(XTest)
    with open(my_file1, 'wb') as f:
        pickle.dump([minor_eclf, major_eclf, Countvec, sk],f)
    return  subCatIdPredictLS, subCatclfLS, parentCatIdPredictLS,parentCatclfLS

def getAllClassifiersPredictions(questionList):

    ensemMethod = "average1"

    if ensemMethod == "average1":
        weight = [1.01, 1.2, 1.09,0.8]


        subCatIdPredictLS, subCatclfLS, parentCatIdPredictLS,parentCatclfLS =  getVotingClassifierPrecitions(questionList)
        #[X, parentCatY, subCatY, XTest,Countvec,sk] = getfeatures(questionList, name="TFIDF")
        predictions = []
        for i in range(subCatIdPredictLS.shape[0]):
            d = {}
            d['minor_category'] = subCatIdPredictLS[i]
            d['confidence_minor_cat'] = subCatclfLS[i].max()

            d['major_category'] = parentCatIdPredictLS[i]
            d['confidence_major_cat'] = parentCatclfLS[i].max()
            predictions.append(d)

        return predictions



    names= ["MNB", "RBFsvm", "Polysvm", "RandomForest" ]
    file = Path("./data/AllEnsemBleClassifiers.npz")
    genNew =False
    if not file.is_file() or genNew:

        parentCatclfLS = []
        subCatclfLS = []
        [X, parentCatY, subCatY, XTest] = getfeatures(questionList, name="bagOfWord")

        subCatclf = MNB(X, subCatY).predict_proba(XTest)
        parentCatclf = MNB(X, parentCatY).predict_proba(XTest)
        subCatclfLS.append(subCatclf)
        parentCatclfLS.append(parentCatclf)

        [X, parentCatY, subCatY, XTest] = getfeatures(questionList, name="TFIDF")
        subCatclf = RBFsvm(X, subCatY).predict_proba(XTest)
        parentCatclf = RBFsvm(X, parentCatY).predict_proba(XTest)
        subCatclfLS.append(subCatclf)
        parentCatclfLS.append(parentCatclf)

        subCatclf = Polysvm(X, subCatY).predict_proba(XTest)
        parentCatclf = Polysvm(X, parentCatY).predict_proba(XTest)
        subCatclfLS.append(subCatclf)
        parentCatclfLS.append(parentCatclf)

        [X, parentCatY, subCatY, XTest] = getfeatures(questionList, name="ngram")
        subCatclf = myRandomForest(X, subCatY).predict_proba(XTest)
        parentCatclf = myRandomForest(X, parentCatY).predict_proba(XTest)
        subCatclfLS.append(subCatclf)
        parentCatclfLS.append(parentCatclf)

        with open(file, 'wb') as f:
            pickle.dump([subCatclfLS, parentCatclfLS], f)
        #s = pickle.dumps([subCatclfLS, parentCatclfLS])
        #np.save(file, [subCatclfLS, parentCatclfLS])

    print("loading existed file {}".format(file))
    #[subCatclfLS, parentCatclfLS] = np.load(file)
    subCatclfLS=[]
    parentCatclfLS=[]
    with open(file, 'rb') as f:
        [subCatclfLS, parentCatclfLS] = pickle.load(f)
        return subCatclfLS, parentCatclfLS
    rule = "average"
    #rule = "prduct"
    threshold = 0.01
    predictions = []
    d = {}
    if ensemMethod ==  "average":
        weight = [1.01, 1.2, 1.09,0.8]

        for _ in parentCatclfLS:
            p = np.zeros((1, len(parentCatclfLS)))
            count = 0
            for clf in subCatclfLS:
                pr = np.array(clf)
                if pr.max() >= threshold:
                    count += 1
                    p += pr
            pr /= count
            yindex = pr.argmax()
            d['minor_category'] = subCatY[yindex]
            d['confidence_minor_cat'] = pr.max()

            p = np.zeros((1, len(parentCatclfLS)))
            count = 0
            for clf in parentCatclfLS:
                pr = np.array(clf)
                if pr.max() >= threshold:
                    count += 1
                    p += pr
            pr /= count
            yindex = pr.argmax()
            d['major_category'] = parentCatY[yindex]
            d['confidence_major_cat'] = pr.max()
            predictions.append(d)
        return predictions

    if ensemMethod == "productRule":
        weight = [1.01, 1.2, 1.09,0.8]
        for _ in parentCatclfLS:
            p = np.zeros((1, len(parentCatclfLS)))
            count = 0
            for clf in subCatclfLS:
                pr = np.array(clf)
                if pr.max() >= threshold:
                    count += 1
                    p += pr
            pr /= count
            yindex = pr.argmax()
            d['minor_category'] = subCatY[yindex]
            d['confidence_minor_cat'] = pr.max()

            p = np.zeros((1, len(parentCatclfLS)))
            count = 0
            for clf in parentCatclfLS:
                pr = np.array(clf)
                if pr.max() >= threshold:
                    count += 1
                    p += pr
            pr /= count
            yindex = pr.argmax()
            d['major_category'] = parentCatY[yindex]
            d['confidence_major_cat'] = pr.max()
            predictions.append(d)
        return predictions







# ***************************** main program *******************************************
if __name__ == "__main__":
    from predict_question_category import *
    questionList =  getQuestionList(qfile = 'question_test.csv')


    print(getAllClassifiersPredictions(questionList))