'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # tf-idf representation
    tf_idf_vectorize = TfidfTransformer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data) #bag-of-word features for input
    #feature_names = tf_idf_vectorize.get_feature_names()
    tf_idf_test = tf_idf_vectorize.transform(test_data)
    return tf_idf_train, tf_idf_test#, feature_names

def bnb_baseline(bow_train, train_labels, bow_test, test_labels,feature_extraction='bow'):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy - {} = {}\n'.format(feature_extraction,(train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy - {} = {}\n'.format(feature_extraction,(test_pred == test_labels).mean()))

    return model

def lr_run(train_data,train_labels,test_data,test_labels,feature_extraction='bow'):

    lr = LogisticRegression();
    lr.fit(train_data,train_labels)


    #TODO
    #hyper-param tuning

    # evaluate the logistic regression model
    train_pred = lr.predict(train_data)
    print('Logistic Regression train accuracy - {} = {}\n'.format(feature_extraction, (train_pred == train_labels).mean()))
    test_pred = lr.predict(test_data)
    print('Logistic Regression train accuracy - {} = {}\n'.format(feature_extraction, (test_pred == test_labels).mean()))

    return lr

def svm_run(train_data,train_labels,test_data,test_labels,feature_extraction='bow'):

    SVM = svm.LinearSVC()
    SVM.fit(train_data, train_labels)

    # TODO
    # hyper-param tuning

    # evaluate the logistic regression model
    train_pred = SVM.predict(train_data)
    print('SVM Regression train accuracy - {} = {}\n'.format(feature_extraction,(train_pred == train_labels).mean()))
    test_pred = SVM.predict(test_data)
    print('SVM Regression train accuracy - {} = {}\n'.format(feature_extraction, (test_pred == test_labels).mean()))

    return SVM

def gnb_run(train_data,train_labels,test_data,test_labels,feature_extraction='bow'):
    gnb = GaussianNB();
    gnb.fit(train_data, train_labels)

    # TODO
    # hyper-param tuning
    # evaluate the logistic regression model
    train_pred = gnb.predict(train_data)
    print('Logistic Regression train accuracy - {} = {}\n'.format(feature_extraction,(train_pred == train_labels).mean()))
    test_pred = gnb.predict(test_data)
    print('Logistic Regression train accuracy - {} = {}\n'.format(feature_extraction, (test_pred == test_labels).mean()))

    return gnb

def dnn(train_data,train_labels,test_data,test_labels,feature_extraction='bow'):
    #TODO
    pass

if __name__ == '__main__':
    train_data, test_data = load_data()
    #print (set(train_data.target))
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)

    bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)

    train_tfidf, test_tfidf = tf_idf_features(train_bow, test_bow)
    #print (train_bow.toarray().shape)
    #print (feature_names.shape)
    bnb_model = bnb_baseline(train_tfidf, train_data.target, test_tfidf, test_data.target,'tf-idf')
    lr_model = lr_run(train_tfidf,train_data.target,test_tfidf,test_data.target,'tf-idf')

    #TODO
    #svm_model = svm_run(train_tfidf,train_data.target,test_tfidf,test_data.target,'tf-idf')
    train_dense = train_tfidf.todense();
    test_dense = test_tfidf.todense();

    gnb_model = gnb_run(train_dense,train_data.target,test_dense,test_data.target,'tf-idf')

