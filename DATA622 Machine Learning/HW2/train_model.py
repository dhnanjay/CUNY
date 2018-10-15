"""
CUNY DATA 622-2 Assignment 2

@author : Dhananjay Kumar
"""
from pull_data import *
import pandas as pd
import numpy as np
import time
import itertools
import sys
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from pickle import dump
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
#figure(num=None, figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')

def get_time(start, end, name):
    """
    This function will return time consumed to run any function and name of the function from where it is invoked
    :param start: Start Time at which the first call to the function was made
    :param end:  Time at which the processing of the function ended
    :param name: Name of the Function from which this function is called
    :return: Null
    """
    print("Function :",name," ||","Seconds :",round((end-start),2),"| ","Minutes :", round((end-start)/60,2))



def dropFeatures(df):
    """
    The function will drop unwanted features from the dataframe. In this case, columns : Name, Ticket No.,Cabin
    seems unwanted.
    :param df: dataframe with above columns
    :return df: dataframe without unwanted columns
    """
    df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    return df

def fixMissingData(df, threshold_perc=25, dummies=10):
    """
    This function will list all the columns that has Missing Values. It will then identify datatype of each column
    which has missing values and will then try to fix it. For the column datatype: int & float whose missing value is
    less than threshold_perc of the total data, it will substitute with mean value. For column datatype: Object with  missing
    value less than threshold_perc & non unique values less than dummies, it will substitute with  Mode value
    :param df: dataframe with missing values
    :param threshold_perc: Value before which data would be fixed automatically.Above this value data needs to be analysed further
    :param dummies: Value below which data is considered fit to be for one hot encoding i.e dummies & hence can be fixed
    :return df: Dataframe with replaced values for Missing data
    """
    ls = df.isnull().sum()
    for idx, x in ls.iteritems():
        if x > 0:
            # List Column Name, No. of data point missing & data missing in %
            print (idx, ":", x, "|", "Missing data in % :", round((int(x) / int(df.shape[0]) * 100), 2))
            if df[idx].dtype == 'float64' or df[idx].dtype == 'int64':
                # For numeric datatypes, if the missing data is less than threshold_perc, replace it with mean
                if round((int(x) / int(df.shape[0]) * 100), 2) < threshold_perc:
                    df[idx].fillna(df[idx].mean(), inplace=True)
                    print("Missing Values in " + idx + " fixed.")
            elif df[idx].dtype == 'object':
                # For non numeric dtype with less than threshold_perc missing values & non unique value less than dummies, replace it with mode
                if round((int(x) / int(df.shape[0]) * 100), 2) < threshold_perc and int(df[idx].nunique()) < dummies:
                    df[idx].fillna(df[idx].mode()[0], inplace=True)
                    print("Missing Values in " + idx + " fixed.")
    return df

def createDummies(df,target='Survived',dummies=10):
    """
    This function will create dummies for any column whose unique value is less than dummies.
     Target variable will be ignored
    :param df: dataframe for which dummies needs to be created
    :param target: target variable needs to be ignored
    :param dummies: Value below which data is considered fit to be for one hot encoding i.e dummies
    :return df:dataframe with dummies
    """
    for column in df:
        if column==target:
            continue
        if int(df[column].nunique()) < dummies:
            df = pd.get_dummies(df, columns=[column])
            print("Dummies created for :", column)
    return df

def rescaleFeatures(df, ignore=['Survived', 'PassengerId'], threshold=9):
    """
    This function will rescale any continuous column data point which was not addressed by create dummies function.
    It will choose only those columns with dtype int or float and whose unique values are more than threshold.
    For rescaling,MinMax scaler is used as it is good for even those distributions which are not normal
    :param df: df whose columns needs to be rescaled
    :param ignore: Columns that needs to be ignored
    :param threshold: Threshold for no. of non unique values in a column above which the column is fit for rescaling.
    :return df: df with rescaled columns
    """
    scaler = MinMaxScaler()
    rescaleCol=[]
    for column in df:
        if column in ignore:
            continue
        if int(df[column].nunique()) > threshold and (df[column].dtype == 'float64' or df[column].dtype == 'int64'):
            rescaleCol.append(column)
    if len(rescaleCol) > 0:
        df[rescaleCol] = scaler.fit_transform(df[rescaleCol])
        print("Following Columns rescaled using MinMaxScaler :", rescaleCol)
    else:
        print(" No column found for rescaling !")
    return df


def tuneHP(df,cv=10,test_size=0.2):
    """
    This func. is used for Hyperparameter tuning. It uses GridSearch for finding optimum combination of Hyperparameters
    :param df: dataframe with data
    :param cv: Cross Validation split, Default Size: 10
    :param test_size: Test Size, Default Size: Test: 20%
    :return:
    """
    startT = time.time()

    # Create arrays for the features and the response variable
    y = df.Survived.values
    X = df.drop(['PassengerId', 'Survived'], axis=1).values
    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # Create Random Forest Model
    rfc = RandomForestClassifier(n_jobs=-1, oob_score=True, random_state=42)

    # Use a grid over parameters of interest
    param_grid = {
        "n_estimators":np.arange(10, 50, 2).tolist(), # list(range(10, 100)),
        "max_depth": np.arange(1, 40, 2).tolist(), #list(range(1,50)),
        "min_samples_leaf": np.arange(1, 10, 2).tolist(), #list(range(1,10)),
        "criterion": ['gini', 'entropy'],
        "max_features": ['sqrt','auto','log2']}

    kfolds = StratifiedKFold(cv)

    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid,verbose=3, scoring='accuracy', # roc_auc
				cv=kfolds.split(X_train,y_train), n_jobs=-1)
                      #    cv=StratifiedKFold(y_train, n_folds=cv, shuffle=True),n_jobs=-1)
    CV_rfc.fit(X_train, y_train)
    print(CV_rfc.best_params_)
    get_time(startT, time.time(), sys._getframe().f_code.co_name)
    return CV_rfc.best_params_

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



def randomForestClassifier(df,n_estimators,features='sqrt',max_depth=10, min_samples_leaf=1,
                           criterion='gini', kFold=10,test_size=0.25):
    """
    This function applies Random Forest Algorithm to the df. It tells AUC for the total dataset.
    Accuracy for the test data from training data set and Confusion Matrix
    :param df:
    :param no_trees:
    :param features:
    :param kFold:
    :param test_size:
    :return:
    """
    startT = time.time()
    # Create arrays for the features and the response variable
    y = df.Survived.values
    X = df.drop(['PassengerId', 'Survived'], axis=1).values

    # Save Model Features in CSV
    with open("model_features.csv", "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in list(df.drop(['PassengerId', 'Survived'], axis=1)):
            writer.writerow([val])
    if os.path.exists('model_features.csv'):
        print("Model Features downloaded")
    else:
        print("Unable to download Model Features")

    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    kfold = KFold(kFold, random_state=42)
    model = RandomForestClassifier(n_jobs=-1, oob_score=True, n_estimators=n_estimators, max_features=features,
                                   max_depth=max_depth, min_samples_leaf=min_samples_leaf,criterion=criterion,random_state=42)
    results = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc')
    # print("Cross Validation Accuracy & SD: %.3f%% (%.3f%%)" % (results.mean() * 100.0, results.std() * 100.0))
    print("AUC & SD: %.3f%% (%.3f%%)" % (results.mean() * 100.0, results.std() * 100.0))
    model.fit(X_train, y_train)
    result = model.score(X_test, y_test)
    print("Accuracy : %.3f%%" % (result * 100.0))

    predicted = model.predict(X_test)
    matrix = confusion_matrix(y_test, predicted)
    print("Confusion Matrix :-")
    print(matrix)
    plt.figure()
    plot_confusion_matrix(matrix, normalize=False,classes=[0,1],title="Confusion Matrix")
    plt.show()

    # save the model to disk
    filename = 'final_model.pkl'
    dump(model, open(filename, 'wb'))
    print("Model saved to local disk")
    get_time(startT, time.time(), sys._getframe().f_code.co_name)

