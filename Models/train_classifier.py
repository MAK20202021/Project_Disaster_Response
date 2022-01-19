# import libraries
import sys
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet'])
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle
from sqlalchemy import create_engine
import sqlite3


def load_data(database_filepath):
    ''' 
    Input: Load data from SQL file
    Return: messaes as X and possible output as Y 
    and name of all categories
    '''
    df = pd.read_sql_table('Response_Table', 'sqlite:///{}'.format(database_filepath))  
    X = df.message.values
    Y = df.iloc[:,4:]
    category_names = list(Y.columns.values)
    return X,Y,category_names


def tokenize(text):
    ''' 
    Input: All text messages
    Return: Cleaned text messages after lemmatizing and normalizing
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens  

def build_model():
    ''' 
    Building Machine Learning Model
    Return: GridSearching parameters
    '''
    pipeline = Pipeline([
        ('vect',CountVectorizer()),
        ('tfidf',TfidfTransformer()),        
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {'tfidf__use_idf':[True, False],
              'clf__estimator__min_samples_leaf': [1, 2],
              'clf__estimator__min_samples_split': [2 , 3]
              }
    cv = GridSearchCV(pipeline, param_grid = parameters )
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    ''' 
    Evaluating Machine Learning Model
    Input: All testing data, building model and all names of categories
    '''
    y_pred = model.predict(X_test)
    print(classification_report(Y_test.iloc[:,1:].values, np.array([x[1:] for x in y_pred]),target_names=category_names))

def save_model(model, model_filepath):
    ''' 
    Saving model into a pickle file
    Input: building model and name of pickle file from user
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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