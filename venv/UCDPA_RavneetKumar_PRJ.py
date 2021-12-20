# Import all the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from xgboost import XGBRegressor
import utils as ut
# Suppress any warnings
import warnings
warnings.filterwarnings("ignore")



if __name__ == '__main__':
    
    # 1. Read csv data into pandas dataframes
    
    calories_data = ut.read_file(src_type='csv', name='calories.csv')
    exercise_data = ut.read_file(src_type='csv', name='exercise.csv')
    
    # 2. Read the same files into pandas dataframe from MySQL database
    #calories_data = ut.read_file(src_type='db', name='calories')
    #exercise_data = ut.read_file(src_type='db', name='exercise')
    
    # 3. Print datatype of the above variables
    print(ut.get_datatype(calories_data))
    print(ut.get_datatype(exercise_data))
    
    # 4. Check which columns contains NULLS
    print(ut.check_nulls(calories_data))
    print(ut.check_nulls(exercise_data))
    
    # 5. TODO: Impute NA values with mean values
    ut.fill_null_values(calories_data, fill_with= 'mean')
    ut.fill_null_values(exercise_data, fill_with= 'mean')
    
    
    # 6. Check the datatype of the columns in the df
    print(ut.check_dtypes(calories_data))
    print(ut.check_dtypes(exercise_data))
    
     # Find and drop duplicated from the dataset
    ut.drop_duplicates(exercise_data)
    ut.drop_duplicates(calories_data)
    
    # 7. Merge datasets using USER ID columns
    data_merged = calories_data.merge(exercise_data, on = 'User_ID')
    
    # 8. Get info about the merged datset
    print(data_merged.info)
    
    # 9. Get the statistical measure of the data
    print(data_merged.describe)
    
    
    #Insights from the data
    sns.set()
    sns.countplot(data_merged['Gender'])
    #plt.show()
    plt.savefig('Gender distribution.png')
    print('Gender values are categorical and evenly distributed')
    
    sns.displot(data_merged['Age'])
    #plt.show()
    plt.savefig('Age distribution.png')
    print('Age count decreases with age increasing')
    
    sns.displot(data_merged['Height'])
    #plt.show()
    plt.savefig('Height distribution.png')
    print('Distribution of Height among the dataset')
    
    sns.displot(data_merged['Weight'])
    #plt.show()
    plt.savefig('Height Distribution.png')
    print('Distribution of Weight among the dataset')
    
    # derive the correaltion using the heatmap
    correlation_exc=data_merged.corr()
    plt.figure(figsize=(6,8))
    sns.heatmap(correlation_exc, cbar=True, square=True,  cmap='Blues')
    #plt.show()
    plt.savefig('Correlation matrxi.png')
    
    # Box plot to detect outliers
    data_merged[['Age', 'Height', 'Weight', 'Duration',  \
                                 'Heart_Rate', 'Body_Temp']].plot.box(grid='True')
    plt.savefig('Box plot.png')

    # 10. Perform train test split
    X = data_merged.copy()
    X.drop(columns = ['User_ID', 'Calories'], inplace=True)
    y = data_merged['Calories']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    
    print('Size of the Train and Test split dataset are as below:')
    print(X_train.shape, X_test.shape)
    
    # 11. Perform label encoding on the dataset(perform after train test split)
    X_train = ut.label_encoder(X_train, columns = ['Gender'])
    X_test = ut.label_encoder(X_test, columns = ['Gender'])
    
    # 12. Train the model
    model = XGBRegressor()
    print("Training the XGBRegressor model on the train dataset")
    model.fit(X_train,y_train)
    
    # 13. predicting with the test data
    predict_calories=model.predict(X_test)
    print('print the target predicted using test data')
    print(predict_calories)
    
    # 14. Print accuracy score, confusion matrix, and MAE for the model
    
    mae=metrics.mean_absolute_error(y_test, predict_calories)
    print('Mean Absolute Error in predictionis: ', mae)
    
    # Hyper paramter tuning
    params= {
        "learning_rate": [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30], 
        "max_depth": [1, 3, 4, 5, 6, 10, 12,  15, 18], 
        "min_child_weight": [1, 3, 5, 7], 
        "gamma": [0.0, 0.1, 0.2, 0.3, 0.4], 
        "colsample_bytree": [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
        }

    random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=5, \
                                      scoring='r2', n_jobs=-1, cv=5, verbose=3)
    random_search.fit(X_train, y_train)
    print('Best parameters for the model are:-')
    print(random_search.best_estimator_)
    y_hyper_pred = random_search.predict(X_test)
    print(y_hyper_pred)
    hyper_mae = metrics.mean_absolute_error(y_test, y_hyper_pred)
    print('MAE after hyper parameter tuning is: ', hyper_mae)
