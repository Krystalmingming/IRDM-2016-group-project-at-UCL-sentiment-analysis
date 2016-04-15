import math
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

if __name__ == '__main__':

    # open file
    posFile = open('/Users/zxy/Desktop/IRDM group/sentiment_analysis_python-master/data/positive.txt', 'r')
    negFile = open('/Users/zxy/Desktop/IRDM group/sentiment_analysis_python-master/data/negative.txt', 'r')

    # extract features
    posFeatures = []
    negFeatures = []
    for posTweet in posFile:
        posWords = [posTweet.rstrip(), 'pos']
        posFeatures.append(posWords)
    for negTweet in negFile:
        negWords = [negTweet.rstrip(), 'neg']
        negFeatures.append(negWords)

    # selects 3/4 of the features to be used for training and 1/4 to be used for testing
    posCutoff = int(math.floor(len(posFeatures) * 3 / 4))
    negCutoff = int(math.floor(len(negFeatures) * 3 / 4))
    trainSet = posFeatures[:posCutoff] + negFeatures[:negCutoff]
    testSet = posFeatures[posCutoff:] + negFeatures[negCutoff:]

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for (tweet, label) in trainSet:
        train_data.append(tweet)
        train_labels.append(label)
    for (tweet, label) in testSet:
        test_data.append(tweet)
        test_labels.append(label)

    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df = 0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)

    print("finish vectors")

    # Perform classification with SVM, kernel=rbf
    classifier_rbf = svm.SVC()
    t0 = time.time()
    classifier_rbf.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_rbf = classifier_rbf.predict(test_vectors)
    t2 = time.time()
    time_rbf_train = t1-t0
    time_rbf_predict = t2-t1

    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel='linear')
    t0 = time.time()
    classifier_linear.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vectors)
    t2 = time.time()
    time_linear_train = t1-t0
    time_linear_predict = t2-t1

    # Perform classification with SVM, kernel=linear
    classifier_liblinear = svm.LinearSVC()
    t0 = time.time()
    classifier_liblinear.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_liblinear = classifier_liblinear.predict(test_vectors)
    t2 = time.time()
    time_liblinear_train = t1-t0
    time_liblinear_predict = t2-t1

    # Print results in a nice table
    print("Results for SVC(kernel=rbf)")
    print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
    print(classification_report(test_labels, prediction_rbf))
    print("Results for SVC(kernel=linear)")
    print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
    print(classification_report(test_labels, prediction_linear))
    print("Results for LinearSVC()")
    print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
    print(classification_report(test_labels, prediction_liblinear))