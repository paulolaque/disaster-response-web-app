import sys
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from sqlalchemy import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import pickle
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    '''
    Load messages and categories data from SQLite database 
    Arguments:
        database_filepath: path to SQLite database
    Output:
        X: input feature
        Y: categories
        category_names: categories names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages_with_categories', engine)
    X = df.message
    Y = df.iloc[:, 4:]
    category_names = df.columns[4:]
    return X, Y, category_names


def tokenize(text):
    '''
    Tokenize words in a string, lower case all characters,
    remove stop words, lemmetize and remove pontuations. 
    Arguments:
        text: string
    Output:
        clean_tokens: tokenized, case normalized, lemmetized and cleaned words
    '''
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word not in string.punctuation]
    return clean_tokens


def build_model():
    '''
    Build Random Forest model using  
    Arguments: 
            none
    Output:
        cv: model pipeline with gridsearch
    '''
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ])
    parameters = {'tfidf__use_idf':[True, False],
                      'clf__estimator__max_depth': [5, 10, 15],
                      'clf__estimator__n_estimators': [2, 5, 10]}
    cv = GridSearchCV(pipeline, param_grid = parameters,n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Show best parameters for trained model
    Arguments:
        model: trained model
        X_test: input test 
        Y_test: output test
        category_names: categories
    Output:
        print best parameters for trained model
    
    '''
    predictions = model.predict(X_test)
    for i in range(0,len(category_names)):
            print(Y_test.columns[i])
            print(classification_report(Y_test.iloc[:,i], pd.DataFrame(predictions)[i]))


def save_model(model, model_filepath):
    '''
    Save the trained model as pickle with selected parameters 
    Arguments:
        model 
        model_filepath
    Output:
        pickle file of model
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