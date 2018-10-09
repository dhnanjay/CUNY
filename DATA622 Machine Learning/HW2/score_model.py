"""
CUNY MSDS Program, DATA 622, Homework 2
Created: Oct 2018
@author: Dhananjay Kumar
This module loads model from pkl file, imports test data and runs prediction.
"""

# Custom module for loading Titanic data set
import warnings
warnings.filterwarnings('ignore')
from pull_data import *
from train_model import *
import pickle
import pandas as pd

def run_test_data(df=pd.read_csv("test.csv")):
    startT = time.time()
    # Read data
    df_test = df

    # Data Preprocessing
    try:
        df_test = dropFeatures(df_test)
        df_test = fixMissingData(df_test)
        df_test = createDummies(df_test)
        df_test = rescaleFeatures(df_test)
    except:
        raise

    # Load model from the pickle file
    try:
        RFC = pickle.load(open('final_model.pkl', 'rb'))
    except:
        raise

    # Remove passenger ID
    p_df = df_test.drop(['PassengerId'], axis=1)

    # Run prediction
    pred_df = RFC.predict(p_df)

    # Concatenate passenger ID and prediction
    pred_df = pd.concat([test_df['PassengerId'], pd.DataFrame(pred_df)], axis=1)
    pred_df.columns = ['PassengerId', 'Survived']

    # Save results to CSV file for Kaggle submission
    try:
        pred_df.to_csv('kaggle_submission.csv', index=False)
        print("File kaggle_submission.csv saved to local directory" )
    except:
        raise
    get_time(startT, time.time(), sys._getframe().f_code.co_name)