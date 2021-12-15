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
plt.figure(figsize=(6,8))
sns.heatmap(correlation_exc, cbar=True, square=True,  cmap='Blues')
#plt.show()

# Gender data is non-numeric, hence convert this classified data into numerical
exercise_data_merged.replace({'Gender':{'male': 0 , 'female': 1}}, inplace=True)
print(exercise_data_merged.head())

# Divide dataset into observations(X) and target(y)
X=exercise_data_merged.drop(columns=['User_ID','Calories'], axis=1)
print('print first five rows of observations')
print(X.head())
y=exercise_data_merged['Calories']
print('print first five rows of target')
print(y.head())

# Divide the observations/target further into training and test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)
print(X.shape, X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Define model from regressor and train with training dataset
model=XGBRegressor()
model.fit(X_train,y_train)

#predicting with the test data
predict_calories=model.predict(X_test)
print('print the target predicted using test data')
print(predict_calories)

# Validate the error using mean absolute error
mae=metrics.mean_absolute_error(y_test, predict_calories)
print('Mean Absolute Error in prediction', mae)