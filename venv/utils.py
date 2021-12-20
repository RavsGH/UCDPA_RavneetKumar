# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 07:55:22 2021

@author: tanay
"""

import pandas as pd
import sqlalchemy as sa
import pyodbc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def connect_db():
    server = 'localhost'
    engine=sa.create_engine('mysql+mysqldb://root:Ms#56789@127.0.0.1/sakila')
    return engine

def read_file(src_type, name):
    print(f'Reading the {src_type} : {name} into pandas dataframe.')
    if src_type =='csv':
        return pd.read_csv(name)
    elif src_type == 'xls':
        return pd.read_excel(name)
    elif src_type == 'db':
        conn = connect_db()
        sql_ = f"""SELECT * from {name}"""
        return pd.read_sql(sql_, conn)
    

def get_datatype(value):
    print('The data type of {value} is:' + str(type(value)))
    # return type(value)

def top_rows(df, num_rows=5):
    print(f'Top {num_rows} for the  are:')
    return df.head(num_rows)

def check_nulls(df):
    print('Columns containing NULLS are:-')
    return df.columns[df.isna().any()].tolist()

def check_dtypes(df):
    print('The datatypes of the columns are:')
    return df.dtypes

def label_encoder(df, columns):
    le = LabelEncoder()
    df[columns] = df[columns].apply(le.fit_transform)
    return df
    
def mannual_label_encoder(df):
    df.replace({'Gender':{'male': 0 , 'female': 1}}, inplace=True)

def drop_duplicates(df):
    duplicated_values = df.loc[df.duplicated()]
    if not duplicated_values.empty:
        print('There are duplicated values in the dataframe: ', len(duplicated_values.index))
        print('Dropping duplicates now!')
        df.drop_duplicates(inplace=True)
    else:
        print('There are no duplicates in the dataframe.')
        
        
def fill_null_values(df, fill_with= 'mean'):
    cols = df.columns[df.isna().any()].tolist()
    if len(cols)> 0 :
        print('Columns containing nulls are: ',cols)
        for col in cols:
            
            if fill_with == 'mean':
                val = df[col].mean()
                print('Filling null values with mean')
            elif fill_with == 'median':
                val = df[col].median()
                print('Filling null values with median')
            df[col].fillna(val, inplace=True)
            
    else:
        print('There are no NULL values in the dataset.')