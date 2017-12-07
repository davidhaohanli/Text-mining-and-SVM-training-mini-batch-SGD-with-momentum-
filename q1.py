'''
Question 1 Full Code

main_best_param() version

main() with hyper-parameters tuning

may take more than 40 min to run the main function

'''
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
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

def cm(test_labels,test_pred):
    CM = np.zeros((20,20));
    for i in range(test_labels.shape[0]):
        CM[int(test_pred[i])][int(test_labels[i])]+=1
    return CM

def cm_f1_test(model,test_data,test_labels):

    test_pred=model.predict(test_data);
    CM = cm(test_labels,test_pred)
    two_confused_mat = CM+CM.T
    np.fill_diagonal(two_confused_mat,0)
    flatten = two_confused_mat.flatten()
    argSort = flatten.argsort()
    argTwoD = (argSort[-1]%20,argSort[-1]//20)
    return CM,(argTwoD,two_confused_mat[argTwoD])

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

    knn = grid.best_estimator_

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

    mnb = grid.best_estimator_
    train_pred = mnb.predict(train_data)
    print('Multinomial Naive Bayes train accuracy - {} = {}\n'.format(feature_extraction,(train_pred == train_labels).mean()))
    test_pred = mnb.predict(test_data)
    accuracy=(test_pred == test_labels).mean()
    print('Multinomial Naive Bayes test accuracy - {} = {}\n'.format(feature_extraction,accuracy))

    return mnb,accuracy

def main():

    train_data, test_data = load_data()
    #train_data.data, train_data.target, test_data.data,test_data.target = train_data.data[:100], train_data.target[:100], test_data.data[:100],test_data.target[:100]
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

    bestAccuracy=sorted(acc.items(), key=lambda d: d[0])[-1][0]
    model=acc[bestAccuracy][0]
    print ('\nThe best model is {}, and the corresponding accuracy is {} \n'.format(acc[bestAccuracy][1],bestAccuracy))
    res = cm_f1_test(model, test_tfidf, test_data.target)
    print(pd.DataFrame(res[0],index=test_data.target_names,columns=test_data.target_names))
    print('Most confused 2 classes: {} and {}\n and their corresponding sum of false labels: {}'. \
          format(test_data.target_names[res[1][0][0]], test_data.target_names[res[1][0][1]], res[1][1]))

def model_run(model,train_data,train_labels,test_data,test_labels,modelName):

    model.fit(train_data, train_labels)
    train_pred = model.predict(train_data)
    print('{} train accuracy = {}\n'.format(modelName, (train_pred == train_labels).mean()))
    test_pred = model.predict(test_data)
    accuracy = (test_pred == test_labels).mean()
    print('{} test accuracy = {}\n'.format(modelName, accuracy))

    return accuracy

def main_best_param():
    train_data, test_data = load_data()
    #train_data.data, train_data.target, test_data.data,test_data.target = train_data.data[:100], train_data.target[:100], test_data.data[:100],test_data.target[:100]
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)
    train_tfidf, test_tfidf = tf_idf_features(train_bow, test_bow)
    acc={}
    models={'BernoulliNB':BernoulliNB(alpha=1.0000000000000001e-05),'LogisticRegression':LogisticRegression(C=2.9000000000000004),\
            'MultinomialNB':MultinomialNB(alpha = 0.022229964825261943),'SVM':svm.SVC(kernel='linear',C=1.4100000000000001)}
    for name,model in models.items():
        accuracy=model_run(model,train_tfidf,train_data.target,test_tfidf,test_data.target,name)
        acc[accuracy] = [model, name]

    bestAccuracy = sorted(acc.items(), key=lambda d: d[0])[-1][0]
    model = acc[bestAccuracy][0]
    print('\nThe best model is {}, and the corresponding accuracy is {} \n'.format(acc[bestAccuracy][1], bestAccuracy))
    res = cm_f1_test(model, test_tfidf, test_data.target)
    print(pd.DataFrame(res[0], index=test_data.target_names, columns=test_data.target_names))
    print('Most confused 2 classes: {} and {}\n and their corresponding sum of false labels: {}'. \
          format(test_data.target_names[res[1][0][0]], test_data.target_names[res[1][0][1]], res[1][1]))

if __name__ == '__main__':
    print ('''

        Notice:

        Running default: main_best_param()

        If you want to run the full code with hyper-parameter tuning (PCA transformation excluded)

        You can change it to main()

        Please be aware that this version may take more than 40 min to run on CPU

        ''')
    main_best_param()