# Import all the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Import the both the datasets (Calories.csv and Excercise.csv)
calories_burnt=pd.read_csv('calories.csv') ###check if absolute path is needed or DB connection
exercise_data = pd.read_csv('exercise.csv')

# check the type of data frames
print('Data type of calories_burnt', type(calories_burnt))
print('Data type of exercise_data', type(exercise_data))
print('#####################################')
# Analyse and read first five rows of the both datasets
print(calories_burnt.head())
print('#####################################')
print(exercise_data.head())

#Merging two data sets for better handling and Analysis
exercise_data_merged=pd.concat([exercise_data,calories_burnt['Calories']],axis=1)
print('#####################################')
print(f'First Five rows of the merged dataset \n',exercise_data_merged.head())
print('#####################################')
print('Rows and Columns', exercise_data_merged.shape)
print('#########Detaisl of the dataset###########')
print(exercise_data_merged.info())

#Check the null values in the dataset
ISNULL=exercise_data_merged.isnull().sum()
print(f'There are below null values in dataset \n',ISNULL)

#Get the statistical measure of the data
exercise_data_merged_stats=exercise_data_merged.describe()
print(f'stats of the merged data \n',exercise_data_merged_stats)

#Insights from the data
sns.set()
sns.countplot(exercise_data_merged['Gender'])
#plt.show()
print('Gender values are categorical and evenly distributed')

sns.displot(exercise_data_merged['Age'])
#plt.show()
print('Age count decreases with age increasing')

sns.displot(exercise_data_merged['Height'])
#plt.show()
print('Distribution of Height among the dataset')

sns.displot(exercise_data_merged['Weight'])
#plt.show()
print('Distribution of Weight among the dataset')

# derive the correaltion using the heatmap
correlation_exc=exercise_data_merged.corr()
plt.figure(figsize=(8,8))
sns.heatmap(correlation_exc, cbar=True, square=True,  cmap='Blues')
plt.show()