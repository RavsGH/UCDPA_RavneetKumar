# Import all the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Import the both the datasets (Calories.csv and Excercise.csv)
calories_burnt=pd.read_csv('calories.csv')
