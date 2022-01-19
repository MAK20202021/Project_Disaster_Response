import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sqlite3

def load_data(messages_filepath, categories_filepath):
    '''
    Input: Files path for Datasets
    Return: Combined dataset after initial cleaning
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #merging both datasets
    df = messages.merge(categories, on='id')
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0].tolist()
    category_colnames = list(map(lambda x: x[:-2], row))
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # Converting all 2 values into 1 in related column
    length = categories.shape[0]
    for i in range(0,length):
        if (categories['related'][i]==2):
            categories['related'][i]=1
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories],axis = 1,sort=True,join = 'inner')
    return df


def clean_data(df):
    ''' 
    Drop duplicates
    Input: Merged Dataframe
    Return: Cleaned Dataframe
    '''
    df.drop_duplicates(subset="id", inplace=True)
    return df


def save_data(df, database_filename):
    ''' 
    Saving data into SQL file
    Input: Dataframe and the fileanme in which data will be saved
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Response_Table', con=engine, index=False, if_exists='replace') 


def main():
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