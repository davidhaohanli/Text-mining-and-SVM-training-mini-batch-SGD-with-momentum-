'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB

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
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names

def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

if __name__ == '__main__':
    train_data, test_data = load_data()
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)

    bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)
