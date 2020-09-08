import sys
# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import pickle
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from scipy.stats import hmean
from scipy.stats.mstats import gmean
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import create_engine




def load_data(database_filepath):
    '''
    Loads data from database
    Args:
        database_filepath: path to database
    Returns:
         X: feature
         Y: labels
         category_name: the names of the categories
    '''
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_response_clean', con = engine)
    
    #Get features and labels
    X = df.message.values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    
    category_names = Y.columns
    
    return X, Y, category_names




def tokenize(text):
    
    '''Tokenize, lemmatize, and normalize text string
    
    Args: text-string containing the message to be processes
    
    Returns: clean_tokens-list of normalized, tokenized strings
    
    '''
    
    # Convert text to lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #tokenize the text
    tokens = word_tokenize(text)
    
    #Initiate a lemmatizer and stopwords remover
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")

    
    #
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return clean_tokens



def build_model():
    
    '''
    
    A machine learning pipeline takes in the message column as input and 
    outputs classification results on the other 36 labels. 
    
    Inputs:  None
   
    Returns:
    cv: a model that uses the message column to predict classifications for 36 categories
    
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    
    parameters = {
            'tfidf__use_idf':[True, False],
            'clf__estimator__n_estimators':[10, 25], 
            'clf__estimator__min_samples_split':[2, 5]
    }
    
    
    
    cv = GridSearchCV(pipeline, param_grid = parameters, n_jobs=4)
    
    return cv

def get_eval_metrics(actual, predicted, cols):
    """Calculates evaluation metrics for the above model
    
    Args:
        actual: An array that contains actual labels.
        predicted: An array containing the predictions.
        cols: List of strings containing names for each of the predicted fields.
       
    Returns:
    metrics_df: dataframe. Dataframe containing the accuracy, precision, recall 
    and f1 score for a given set of actual and predicted labels.
    """
    performance_metrics = []
    
    # Calculate evaluation metrics for each set of labels
    for i in range(len(cols)):
        accuracy = accuracy_score(actual[:, i], predicted[:, i])
        precision = precision_score(actual[:, i], predicted[:, i], average = 'weighted')
        recall = recall_score(actual[:, i], predicted[:, i], average = 'weighted')
        f1 = f1_score(actual[:, i], predicted[:, i], average = 'weighted')
        
        performance_metrics.append([accuracy, precision, recall, f1])
    
    # Create dataframe with the metrics metrics
    metrics_arr = np.array(performance_metrics)
    metrics_df = pd.DataFrame(data = metrics_arr, index = cols, columns = ['Accuracy', 'Precission', 'Recall', 'f1'])
      
    print(metrics_df)



def save_model(model, model_filepath):
    '''
    
    Save the model to a specified path
    
    Parameters:
    model: Machine learning model
    model_filepath: the file path that the model will be saved
    Returns:
    None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    Main function that train the classifier
    Parameters:
    arg1: the file path of the database
    arg2: the file path that the trained model will be saved
    Returns:
    None
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        #Get the predictions
        Y_test_preds = model.predict(X_test)
        
        #list of the predictions
        cols_data = list(Y.columns.values)

        
        print('Evaluating model...')
        get_eval_metrics(np.array(Y_test), Y_test_preds, cols_data)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()