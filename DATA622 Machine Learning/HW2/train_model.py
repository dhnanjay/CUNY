"""
CUNY DATA 622-2 Assignment 2

@author : Dhananjay Kumar
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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

def fixMissingData(df):
    """
    This function will list all the columns that has Missing Values. It will then identify datatype of each column
    which has missing values and will then try to fix it. For the column datatype: int & float whose missing value is
    less than 25% of the total data, it will substitute with mean value. For column datatype: Object with  missing
    value less than 25% & non unique values less than 6, it will substitute with  Mode value
    :param df: dataframe with missing values
    :return df: Dataframe with replaced values for Missing data
    """
    ls = df.isnull().sum()
    for idx, x in ls.iteritems():
        if x > 0:
            # List Column Name, No. of data point missing & data missing in %
            print (idx, ":", x, "|", "Missing data in % :", round((int(x) / int(df.shape[0]) * 100), 2))
            if df[idx].dtype == 'float64' or df[idx].dtype == 'int64':
                # For numeric datatypes, if the missing data is less than 25%, replace it with mean
                if round((int(x) / int(df.shape[0]) * 100), 2) < 25:
                    df[idx].fillna(df[idx].mean(), inplace=True)
                    print("Missing Values in " + idx + " fixed.")
            elif df[idx].dtype == 'object':
                # For non numeric dtype with less than 25 % missing values & non unique value less than 6, replace it with mode
                if round((int(x) / int(df.shape[0]) * 100), 2) < 25 and int(df[idx].nunique()) < 6:
                    df[idx].fillna(df[idx].mode()[0], inplace=True)
                    print("Missing Values in " + idx + " fixed.")

    return df

def createDummies(df):
    """
    This function will create dummies for any column which has 5 or less unique values
    :param df: dataframe for which dummies needs to be created
    :return df:dataframe with dummies
    """
    for column in df:
        if int(df[column].nunique) < 6:
            df = pd.get_dummies(df, columns=[column])
            print("Dummies created for :", column)
    return df

def rescaleFeatures(df):
    """
    This function will rescale any continuous column data point which was not addressed by create dummies function.
    It will choose only those columns with dtype int or float and whose unique values are more than 5. For rescaling,
    MinMax scaler is used as it is good for even those distributions which are not normal
    :param df: df whose columns needs to be rescaled
    :return df: df with rescaled columns
    """
    scaler = MinMaxScaler()
    for column in df:
        if int(df[column].nunique) > 5 and (df[column].dtype == 'float64' or df[column].dtype == 'int64'):
            df[column] = scaler.fit_transform(df[column])
            print("Column " + column + ' rescaled')
    return df





def randomForestClassifier(df,no_trees=100,features=12,kFold=10,test_size=0.3,stratify=y):
    # Create arrays for the features and the response variable
    y = df.Survived.values
    X = df.drop(['PassengerId','Survived'], axis=1).values
    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)

    kfold = KFold(kFold, random_state=42)
    model = RandomForestClassifier(n_estimators=no_trees, max_features=features)
    results = cross_val_score(model, X, y, cv=kfold)
    print(results.mean())
    model.fit(X_train, y_train)
    result = model.score(X_test, y_test)
    print("Accuracy: %.3f%%" % (result * 100.0))

    predicted = model.predict(X_test)
    matrix = confusion_matrix(y_test, predicted)
    print(matrix)

    # save the model to disk
    filename = 'final_model.sav'
    dump(model, open(filename, 'wb'))
