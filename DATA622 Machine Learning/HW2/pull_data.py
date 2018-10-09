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

def downloadTitanic(mode=1,username="UserName",password="Password"):
    """
    !!!    Please Read this  !!!
    This function will download Titanic Training & Test Data. To download, this function will use Kaggle API through
     Kaggle Python package. Authentication will be taken care by Kaggle using Kaggle.json file. Benefit of using
     Kaggle API is that the user need not mention Login Credentials and API key explicitly thus improving security and
     collaborative aspect of the assignment. For more information refer to :
     https://github.com/Kaggle/kaggle-api
    :param mode: 1 -> Will use this function and will not need Username or Password,
           mode != 1 -> Will use downloadTitanicOld() function & will use Username & Password
    :param username: Optional, only needed when mode != 1
    :param password: Optional, only needed when mode != 1
    :return success: 0 -> Download Error, 1 -> Download Successful
    """
    startT = time.time()
    success=0
    if mode==1:
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
    else:
        success=downloadTitanicOld(username,password)

    get_time(startT, time.time(), sys._getframe().f_code.co_name)
    return success

def downloadTitanicOld(username,password):
    """
    !!! Only Use this method if you are unable to use Func downloadTitanic()!!!!
    Deprecated way of Login to Kaggle. where user has to expose username password. Use this method if you are unable to
    configure kaggle.json. This function is a fallback option for users who are not able to use the better
    downloadTitanic() Function. Do not call this function directly. Call function downloadTitanic(mode=2, uname, pswd)
    :param username: Kaggle Username
    :param password: Kaggle Password
    :return success: 0 -> Download Error, 1 -> Download Successful
    """
    startT = time.time()
    success = 0

    train_url = 'https://www.kaggle.com/c/titanic/download/train.csv '

    # The local path where the data set is saved.
    local_filename = "train.csv"

    r = requests.get(train_url)

    # Kaggle Username and Password
    kaggle_info = {'UserName': username, 'Password': password}

    # Login to Kaggle and retrieve the data.
    r = requests.post(r.url, data=kaggle_info, prefetch=False)

    # Writes the data to a local file one chunk at a time.
    f = open(local_filename, 'w')
    for chunk in r.iter_content(chunk_size=512 * 1024):  # Reads 512KB at a time into memory
        if chunk:  # filter out keep-alive new chunks
            f.write(chunk)
    f.close()

    if os.path.exists('train.csv'):
        print("Titanic training file downloaded")
        success = 1
    else:
        print("Unable to download titanic training CSV file !!")
        success = 0

    test_url = 'https://www.kaggle.com/c/titanic/download/test.csv'

    # The local path where the data set is saved.
    local_filename = "test.csv"

    r = requests.get(test_url)

    # Login to Kaggle and retrieve the data.
    r = requests.post(r.url, data=kaggle_info, prefetch=False)

    # Writes the data to a local file one chunk at a time.
    f = open(local_filename, 'w')
    for chunk in r.iter_content(chunk_size=512 * 1024):  # Reads 512KB at a time into memory
        if chunk:  # filter out keep-alive new chunks
            f.write(chunk)
    f.close()

    if os.path.exists('test.csv'):
        print("Titanic training file downloaded")
        success = 1
    else:
        print("Unable to download titanic training CSV file !!")
        success = 0

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

