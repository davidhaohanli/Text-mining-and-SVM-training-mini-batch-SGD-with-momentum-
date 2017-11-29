'''
Question 1 Full Code

with hyper-parameters tuning

It may take more than 40 min to run the main function

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
from sklearn.metrics import f1_score as f1
from sklearn.metrics import confusion_matrix as cm
import pandas as pd

def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test

def doPCA(train_data,test_data):
    from sklearn.decomposition import TruncatedSVD as PCA
    pca=PCA(n_components=5000);
    train_pca=pca.fit_transform(train_data);
    test_pca=pca.transform(test_data);

    return train_pca,test_pca

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer(stop_words='english')
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

def cm_f1_test(model,test_data,test_labels):

    test_pred=model.predict(test_data);
    scores=f1(test_labels, test_pred, average=None)
    argSort=scores.argsort()
    scores=scores[argSort]
    return cm(test_labels,test_pred),(argSort[:2],scores[:2])

def bnb_baseline(bow_train, train_labels, bow_test, test_labels,feature_extraction='TF-IDF'):
    # training the baseline model
    #binary_train = (bow_train>0).astype(int)
    #binary_test = (bow_test>0).astype(int)

    bnb=BernoulliNB()

    alpha_range=np.geomspace(1e-5,1,10)
    param_grid=dict(alpha=alpha_range)
    grid=GridSearchCV(bnb,param_grid,cv=10,scoring='accuracy',n_jobs=-1)
    grid.fit(bow_train,train_labels)
    print('\nThe best hyper-parameter for -- {} is {}, the corresponding mean accuracy through 10 Fold test is {} \n'\
          .format('BernoulliNB',grid.best_params_,grid.best_score_))

    bnb=grid.best_estimator_
    bnb.fit(bow_train, train_labels)
    #evaluate the baseline model
    train_pred = bnb.predict(bow_train)
    print('BernoulliNB baseline train accuracy - {} = {}\n'.format(feature_extraction,(train_pred == train_labels).mean()))
    test_pred = bnb.predict(bow_test)
    accuracy=(test_pred == test_labels).mean()
    print('BernoulliNB baseline test accuracy - {} = {}\n'.format(feature_extraction,accuracy))
    return bnb,accuracy

def lr_run(train_data,train_labels,test_data,test_labels,feature_extraction='TF-IDF'):

    lr=LogisticRegression()

    c_range = np.arange(0.1,3,0.1)
    param_grid = dict(C=c_range)
    grid = RandomizedSearchCV(lr, param_grid, cv=10, scoring='accuracy', n_iter=10, random_state=5,n_jobs=-1)
    grid.fit(train_data, train_labels)
    print('\nThe best hyper-parameter for -- {} is {}, the corresponding mean accuracy through 10 Fold test is {} \n' \
        .format('logistic regression', grid.best_params_, grid.best_score_))

    lr=grid.best_estimator_
    lr.fit(train_data,train_labels)

    train_pred = lr.predict(train_data)
    print('Logistic Regression train accuracy - {} = {}\n'.format(feature_extraction, (train_pred == train_labels).mean()))
    test_pred = lr.predict(test_data)
    accuracy=(test_pred == test_labels).mean()
    print('Logistic Regression test accuracy - {} = {}\n'.format(feature_extraction, accuracy))

    return lr,accuracy

def svm_run(train_data,train_labels,test_data,test_labels,feature_extraction='TF-IDF'):

    SVM = svm.SVC(kernel='linear')

    c_range = np.arange(0.01, 2.01, 0.2)
    param_grid = dict(C=c_range)
    grid = GridSearchCV(SVM, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    grid.fit(train_data, train_labels)
    print('\nThe best hyper-parameter for -- {} is {}, the corresponding mean accuracy through 10 Fold test is {} \n' \
          .format('SVM', grid.best_params_, grid.best_score_))

    SVM=grid.best_estimator_

    SVM.fit(train_data, train_labels)

    # TODO hyper-param tuning

    # evaluate the logistic regression model
    train_pred = SVM.predict(train_data)
    print('SVM train accuracy - {} = {}\n'.format(feature_extraction,(train_pred == train_labels).mean()))
    test_pred = SVM.predict(test_data)
    accuracy=(test_pred == test_labels).mean()
    print('SVM test accuracy - {} = {}\n'.format(feature_extraction, accuracy))

    return SVM,accuracy

def knn_run(train_data,train_labels,test_data,test_labels,feature_extraction='TF-IDF'):
    knn = KNeighborsClassifier()

    k_range = range(1, 100)
    param_grid = dict(n_neighbors=k_range)
    grid = RandomizedSearchCV(knn, param_grid, cv=10, scoring='accuracy', n_iter=10, random_state=5,n_jobs=-1)
    grid.fit(train_data, train_labels)
    print('\nThe best hyper-parameter for -- {} is {}, the corresponding mean accuracy through 10 Fold test is {} \n' \
          .format('KNN', grid.best_params_, grid.best_score_))

    lr = grid.best_estimator_
    knn = KNeighborsClassifier(n_neighbors=10);
    knn.fit(train_data, train_labels)

    train_pred = knn.predict(train_data)
    print('KNN train accuracy - {} = {}\n'.format(feature_extraction,(train_pred == train_labels).mean()))
    test_pred = knn.predict(test_data)
    accuracy=(test_pred == test_labels).mean()
    print('KNN test accuracy - {} = {}\n'.format(feature_extraction,accuracy))

    return knn,accuracy

def mnb_run(train_data,train_labels,test_data,test_labels,feature_extraction='TF-IDF'):

    mnb = MultinomialNB()
    alpha_range = np.geomspace(1e-3, 1, 50)
    param_grid = dict(alpha=alpha_range)
    grid = GridSearchCV(mnb, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    grid.fit(train_data, train_labels)
    print('\nThe best hyper-parameter for -- {} is {}, the corresponding mean accuracy through 10 Fold test is {} \n' \
          .format('MultinomialNB', grid.best_params_, grid.best_score_))

    mnb.fit(train_data, train_labels)
    train_pred = mnb.predict(train_data)
    print('Multinomial Naive Bayes train accuracy - {} = {}\n'.format(feature_extraction,(train_pred == train_labels).mean()))
    test_pred = mnb.predict(test_data)
    accuracy=(test_pred == test_labels).mean()
    print('Multinomial Naive Bayes test accuracy - {} = {}\n'.format(feature_extraction,accuracy))

    return mnb,accuracy

def main():

    train_data, test_data = load_data()
    acc={}

    train_bow, test_bow, feature_names = bow_features(train_data, test_data)
    _,_ = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target,'bow')

    #baseline models
    train_tfidf,test_tfidf = tf_idf_features(train_bow, test_bow)
    bnb_model,bnb_acc = bnb_baseline(train_tfidf, train_data.target, test_tfidf, test_data.target)
    acc[bnb_acc]=[bnb_model,'BernoulliNB']

    #chosen models
    lr_model,lr_acc = lr_run(train_tfidf,train_data.target,test_tfidf,test_data.target)
    mnb_model,mnb_acc = mnb_run(train_tfidf, train_data.target, test_tfidf, test_data.target)
    svm_model,svm_acc = svm_run(train_tfidf,train_data.target,test_tfidf,test_data.target)
    acc[lr_acc]=[lr_model,'LogisticRegression']
    acc[mnb_acc]=[mnb_model,'MultinomialNB']
    acc[svm_acc]=[svm_model,'SVM']

    #not good models
    knn_model,knn_acc = knn_run(train_tfidf,train_data.target,test_tfidf,test_data.target)
    acc[knn_acc]=[knn_model,'KNN']

    bestAccuracy=min(acc,key=acc.get)
    model=acc[bestAccuracy][0]
    print ('\nThe best model is {}, and the corresponding accuracy is {} \n'.format(acc[bestAccuracy][1],bestAccuracy))
    res = cm_f1_test(model, test_tfidf, test_data.target)
    print(pd.DataFrame(res[0],index=test_data.target_names,columns=test_data.target_names))
    print('Most confused 2 classes: {} and {}\n and their corresponding F1 scores: {}'. \
          format(test_data.target_names[res[1][0][0]], test_data.target_names[res[1][0][1]], res[1][1]))


if __name__ == '__main__':
    print ('''

        Notice:

        Q1 with hyper-parameter tuning (PCA transformation excluded)

        This version may take more than 40 min to run

        ''')
    main()