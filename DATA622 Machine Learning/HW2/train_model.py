"""
CUNY DATA 622-2 Assignment 2

@author : Dhananjay Kumar
"""

import pull_data as loader
# Standard required packages
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

def cleanData(df):
    df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)


