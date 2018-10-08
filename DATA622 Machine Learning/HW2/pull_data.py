"""
CUNY DATA 622-2
Assignment No. 2
Author: Dhananjay Kumar
Date: 10/05/2018
"""

import pandas as pd
import os
import re
import requests
import time
import sys
import random
import kaggle


def get_time(start, end, name):
    """
    This function will return time consumed and name of the function invoked
    :param start: Start Time at which the first call to the function was made
    :param end:  Time at which the processing of the function ended
    :param name: Name of the Function from which this function is called
    :return: Null
    """
    print("Function :",name," ||","Seconds :",round((end-start),2),"| ","Minutes :", round((end-start)/60,2))

def downloadTitanic():
    """
    This function will download Titanic Training & Test Data. To download, this function will use Kaggle API through
     Kaggle Python package. Authentication will be taken care by Kaggle using Kaggle.json file. Benefit of using
     Kaggle API is that the user need not mention Login Credentials and API key explicitly thus improving security and
     collaborative aspect of the assignment. For more information refer to :
     https://github.com/Kaggle/kaggle-api
    :return: success -> 0: Download Error, 1: Download Successful
    """
    startT = time.time()
    success=0
    os.system('kaggle competitions download -c titanic -w')
    if os.path.exists('train.csv'):
        print("Titanic training file downloaded")
        success=1
    else:
        print("Unable to download titanic training CSV file !!")
    if os.path.exists('test.csv'):
        print("Titanic test file downloaded")
        success=1
    else:
        print("Unable to download titanic test csv file !!")
        success=0

    get_time(startT, time.time(), sys._getframe().f_code.co_name)
    return success

def loadTitanic():
    """
    This function will call downloadTitanic function to download data from Kaggle, will validate it
    and then will return two dataframes for test & train
    :return: Two dataframe for Train & Test Data
    """
    startT = time.time()
    success=downloadTitanic()
    if success==0:
        print("Download unsuccessful !!")
    else:    # Validate Downloaded Data
        df_train=pd.read_csv("train.csv")
        if df_train.shape[0]<1:
            print("No records found in train.csv")
        else:
            print("No. of records in training data :", df_train.shape)
        df_test = pd.read_csv("test.csv")
        if df_test.shape[0] < 1:
            print("No records found in test.csv")
        else:
            print("No. of records in test data :", df_test.shape)
    get_time(startT, time.time(), sys._getframe().f_code.co_name)
    return(df_train, df_test)

