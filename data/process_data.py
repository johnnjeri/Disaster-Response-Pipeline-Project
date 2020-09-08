import sys
import pandas as pd
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    
    
    '''
    Loads the messages and categories datasets from the specified filepaths
    Args:
        messages_filepath: Filepath to the messages dataset
        categories_filepath: Filepath to the categories dataset
        
    Returns:
        (DataFrame) df: Merged Pandas dataframe
    '''
    
    #load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    #load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    #merging the datasets
    df = pd.merge(messages, categories, on='id')
    
    return df



def clean_data(df):
    """
    Function that cleans the merged dataset
    Args:
        df: Merged pandas dataframe
    Returns:
        df: Cleaned dataframe
    """
    
    #Splitting categories into separate category columns
    # create a dataframe of the 36 individual category columns
    categories =df['categories'].str.split(pat=';', n=-1, expand = True)
    for column in categories.columns:
        categories[column] = categories[column].str.split('-').str[1].astype(int)
        
    
    # select the first row of the categories dataframe
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames=[]
    list_cols = df['categories'].iloc[0].split(';')
    for col in list_cols:
        col = col.split('-')[0]
        category_colnames.append(col)
        
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # drop the original categories column from `df`
    df= df.drop('categories', axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories],axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    Saves clean dataset into an sqlite database
    Args:
        df:  Cleaned dataframe
        database_filename: Name of the database file
    """
    
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('disaster_response_clean', engine, index=False, if_exists='replace')


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