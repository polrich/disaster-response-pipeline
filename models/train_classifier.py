import sys
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import sklearn.externals as extjoblib
import joblib

import pickle

def load_data(database_filepath):
    '''
    Input 
    ----------
    database_filepath

    Output
    ----------
    Features X
    Responses Y
    category_names
    
    This functions takes the database_filepath as input and returns the dataframes
    of features and responses and the response names 
    '''
    
    print('sqlite:///{}'.format(database_filepath))
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', con=engine) 
    print(df.head())
    
    X = df['message']
    Y = df[df.columns[4:]]
    category_names = Y.columns
    
    return X, Y, category_names 


def tokenize(text):
    '''   
    Input
    ----------
    text

    Output
    -------
    clean_tokens
    
    This function takes a text as input, here a feature i.e. message, replaces urls
    with placeholders, tokenizes and lemmatizes the text and returns the clean token

    '''
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    This function sets up the model as a sklearn pipeline containing a countvectorizer,
    TfidTransformer and MultiOutputclassifier with RandomForest classifier.
    GridSearchCV is used to find the best parameters for the model
    '''
    pipeline = Pipeline([
        ('cvect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # parameters set to this due to reduce the size of pkl file, which were too large (600MB) for uploading to github with my previous parameters.
    parameters = {
    'clf__estimator__n_estimators': [10, 100],
    'clf__estimator__max_depth': [5, 50], 
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=-1, verbose=3)    
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Input
    ----------
    model 
    X_test 
    Y_test 
    category_names 

    Thos functions shows the f1 score, precision and recall for the test set for each category

    '''
    Y_pred = model.predict(X_test)
    class_report = classification_report(Y_test, Y_pred, target_names=category_names)
    print(class_report)
    
    for idx in range(Y_pred.shape[1]):
        print('=======================',Y_test.columns[idx],'=======================')
        print(classification_report(Y_test.iloc[:,idx], Y_pred[:,idx]))      


def save_model(model, model_filepath):   
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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