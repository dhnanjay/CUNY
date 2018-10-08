"""
CUNY DATA 622-2 Assignment 2

@author : Dhananjay Kumar
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from pickle import dump


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
        print("Following Columns rescaled using MinMaxSxaler :", rescaleCol)
    else:
        print(" No column found for rescaling !")
    return df





def randomForestClassifier(df,no_trees=100,features=12,kFold=10,test_size=0.3):
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
    # Create arrays for the features and the response variable
    y = df.Survived.values
    X = df.drop(['PassengerId','Survived'], axis=1).values
    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    kfold = KFold(kFold, random_state=42)
    model = RandomForestClassifier(n_estimators=no_trees, max_features=features)
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

    # save the model to disk
    filename = 'final_model.sav'
    dump(model, open(filename, 'wb'))
