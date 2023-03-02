import sys
import numpy as np
import pandas as pd
from sqlalchemy import *
import os

def load_data(messages_filepath, categories_filepath):
    '''
    Function to load data from messages 
    and categories in .csv format
    Arguments:
        messages_filepath: path to messages.csv file
        categories_filepath: path to categories.csv file
    Output:
        Load DataFrame
    '''    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on=['id'])
    return df


def clean_data(df):
    '''
    Function to build table structure of loaded
     categories.csv and clean missing data
    Arguments:
        df: Raw DataFrame
    Outputs:
        df: Cleaned DataFrame
    '''
    categories_content=df.categories.str.split(';',expand=True) #split to columns
    col=categories_content.shape[1] 
    for i in range(col): 
        inside = categories_content[i].str.split('-',expand=True)
        feature=inside[0].unique()[0]
        categories_content[i]=categories_content[i].str.replace(feature+'-','')
        if i==0:
            features_list= inside[0].unique()
        else:
            features_list= np.append(features_list,inside[0].unique())
    categories_content.columns=features_list
    categories_content=categories_content.astype(int)  
    categories_content[categories_content>1]=1 # convert to binary
    df.drop(columns='categories',inplace=True)
    df=df.join(categories_content)
    categories_content=None #clean variable
    df.dropna(how='all', axis=1,inplace=True) #drop columns that only have nan values
    df.drop_duplicates(inplace=True) #drop duplicated rows
    return df 


def save_data(df, database_filename):
    '''
    Function to save data to sql database using sqlite engine
    Arguments:
        df: Cleaned DataFrame
        database_filename: Database file to input DataFrame's data 
    Output:
        Load data from dataframe to database
    '''
    os.getcwd()
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages_with_categories', engine, index=False, if_exists='replace')


def main():
    '''
    Execute functions:
    load_data: Load message and categories csv
    clean_data: Clean data
    save_data: save cleaned data to sqlite database
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()