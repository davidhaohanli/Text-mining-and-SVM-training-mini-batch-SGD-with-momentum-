'''
Question 1 Skeleton Code


'''
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

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
    print('Most common word in training set is "{}"\n'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # tf-idf representation
    tf_idf_vectorize = TfidfTransformer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data) #bag-of-word features for input
    #feature_names = tf_idf_vectorize.get_feature_names()
    tf_idf_test = tf_idf_vectorize.transform(test_data)
    return tf_idf_train, tf_idf_test#, feature_names

def bnb_baseline(bow_train, train_labels, bow_test, test_labels,feature_extraction='TF-IDF'):
    # training the baseline model
    #binary_train = (bow_train>0).astype(int)
    #binary_test = (bow_test>0).astype(int)

    bnb=BernoulliNB()

    bin_range=range(0,10)
    param_grid=dict(binarize=bin_range)
    #TODO set n_jobs if parallel comp wanted
    grid=GridSearchCV(bnb,param_grid,cv=10,scoring='accuracy')
    grid.fit(bow_train,train_labels)
    print('\nThe best hyper-parameter for {} -- {} is {}, the corresponding mean accuracy through 10 Fold test is {} \n'\
          .format('BernoulliNB','Binarize',grid.best_params_,grid.best_score_))

    #old
    '''
    model_accuracy=float('-inf')
    for i in range(0,50,5):
        model=BernoulliNB(binarize=i)
        mean_accuracy=cross_val_score(model,bow_train,train_labels,cv=10).mean()
        if mean_accuracy > model_accuracy:
            model_accuracy=mean_accuracy
            bnb=model

    print ('\nBest hyper-parameter for the model --- {} is {}\n'.format('binarize',bnb.binarize))
    '''

    bnb=grid.best_estimator_
    bnb.fit(bow_train, train_labels)
    #evaluate the baseline model
    train_pred = bnb.predict(bow_train)
    print('BernoulliNB baseline train accuracy - {} = {}\n'.format(feature_extraction,(train_pred == train_labels).mean()))
    test_pred = bnb.predict(bow_test)
    print('BernoulliNB baseline test accuracy - {} = {}\n'.format(feature_extraction,(test_pred == test_labels).mean()))

    return bnb

def lr_run(train_data,train_labels,test_data,test_labels,feature_extraction='TF-IDF'):

    lr=LogisticRegression()

    c_range = np.arange(2,0.1,-0.1)
    param_grid = dict(C=c_range)
    grid = RandomizedSearchCV(lr, param_grid, cv=10, scoring='accuracy', n_iter=10, random_state=5)
    grid.fit(train_data, train_labels)
    # TODO set n_jobs if parallel comp wanted
    print('\nThe best hyper-parameter for {} -- {} is {}, the corresponding mean accuracy through 10 Fold test is {} \n' \
        .format('logistic regression','C', grid.best_params_, grid.best_score_))

    lr=grid.best_estimator_
    lr.fit(train_data,train_labels)

    train_pred = lr.predict(train_data)
    print('Logistic Regression train accuracy - {} = {}\n'.format(feature_extraction, (train_pred == train_labels).mean()))
    test_pred = lr.predict(test_data)
    print('Logistic Regression test accuracy - {} = {}\n'.format(feature_extraction, (test_pred == test_labels).mean()))

    return lr

def svm_run(train_data,train_labels,test_data,test_labels,feature_extraction='TF-IDF'):

    SVM = svm.SVC(kernel='linear')
    SVM.fit(train_data, train_labels)

    # TODO
    # hyper-param tuning

    # evaluate the logistic regression model
    train_pred = SVM.predict(train_data)
    print('SVM train accuracy - {} = {}\n'.format(feature_extraction,(train_pred == train_labels).mean()))
    test_pred = SVM.predict(test_data)
    print('SVM test accuracy - {} = {}\n'.format(feature_extraction, (test_pred == test_labels).mean()))

    return SVM

def knn_run(train_data,train_labels,test_data,test_labels,feature_extraction='TF-IDF'):
    knn = KNeighborsClassifier(n_neighbors=10);
    knn.fit(train_data, train_labels)

    # TODO
    # hyper-param tuning
    # evaluate the logistic regression model
    train_pred = knn.predict(train_data)
    print('KNN train accuracy - {} = {}\n'.format(feature_extraction,(train_pred == train_labels).mean()))
    test_pred = knn.predict(test_data)
    print('KNN test accuracy - {} = {}\n'.format(feature_extraction, (test_pred == test_labels).mean()))

    return knn

def dt_run(train_data,train_labels,test_data,test_labels,feature_extraction='TF-IDF'):

    dt = DecisionTreeClassifier();
    dt.fit(train_data, train_labels)

    # TODO
    # hyper-param tuning
    # evaluate the logistic regression model
    train_pred = dt.predict(train_data)
    print('Decision Tree train accuracy - {} = {}\n'.format(feature_extraction,(train_pred == train_labels).mean()))
    test_pred = dt.predict(test_data)
    print('Decision Tree test accuracy - {} = {}\n'.format(feature_extraction, (test_pred == test_labels).mean()))

    return dt

def mnb_run(train_data,train_labels,test_data,test_labels,feature_extraction='TF-IDF'):
    mnb = MultinomialNB();
    mnb.fit(train_data, train_labels)

    # TODO
    # hyper-param tuning
    # evaluate the logistic regression model
    train_pred = mnb.predict(train_data)
    print('Multinomial Naive Bayes train accuracy - {} = {}\n'.format(feature_extraction,(train_pred == train_labels).mean()))
    test_pred = mnb.predict(test_data)
    print('Multinomial Naive Bayes test accuracy - {} = {}\n'.format(feature_extraction, (test_pred == test_labels).mean()))

    return mnb

def gnb_run(train_data,train_labels,test_data,test_labels,feature_extraction='TF-IDF'):
    #TODO too slow
    gnb = GaussianNB();
    gnb.fit(train_data, train_labels)

    # TODO
    # hyper-param tuning
    # evaluate the logistic regression model
    train_pred = gnb.predict(train_data)
    print('Gaussian Naive Bayes train accuracy - {} = {}\n'.format(feature_extraction,(train_pred == train_labels).mean()))
    test_pred = gnb.predict(test_data)
    print('Gaussian Naive Bayes test accuracy - {} = {}\n'.format(feature_extraction, (test_pred == test_labels).mean()))

    return gnb

def dnn(train_data,train_labels,test_data,test_labels,feature_extraction='TF-IDF'):
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
    bnb_model = bnb_baseline(train_tfidf, train_data.target, test_tfidf, test_data.target)

    #chosen models
    lr_model = lr_run(train_tfidf,train_data.target,test_tfidf,test_data.target)
    mnb_model = mnb_run(train_tfidf, train_data.target, test_tfidf, test_data.target)
    svm_model = svm_run(train_tfidf,train_data.target,test_tfidf,test_data.target)

    #not good models
    #knn_model = knn_run(train_tfidf,train_data.target,test_tfidf,test_data.target,'tf-idf')
    #dt_model = dt_run(train_tfidf,train_data.target,test_tfidf,test_data.target,'tf-idf')

    #gnb too slow
    #gnb_model = gnb_run(train_dense,train_data.target,test_dense,test_data.target,'tf-idf')
