import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

# import libraries
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import sqlite3
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
import re
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer, accuracy_score, classification_report
import pickle



def load_data(database_filepath):
    """ This loads the data from the database
    input: database_filepath: path of the database
    output:
            X: messages
            y: labels
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_query("""SELECT * from messages""", engine)
    X = df['message']
    y = df[['related',
        'request',
        'offer',
        'aid_related',
        'medical_help',
        'medical_products',
        'search_and_rescue',
        'security',
        'military',
        'child_alone',
        'water',
        'food',
        'shelter',
        'clothing',
        'money',
        'missing_people',
        'refugees',
        'death',
        'other_aid',
        'infrastructure_related',
        'transport',
        'buildings',
        'electricity',
        'tools',
        'hospitals',
        'shops',
        'aid_centers',
        'other_infrastructure',
        'weather_related',
        'floods',
        'storm',
        'fire',
        'earthquake',
        'cold',
        'other_weather',
        'direct_report']]

    category_names = y.columns.tolist()
    return X, y, category_names
    

def tokenize(text):
    """Function 
    
    Args:
    text: string. The text data is the message that we will be processing
       
    Returns:
    stemmed: list of strings. List containing normalized and processed word tokens
    """
    # Change text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # Tokenize words
    tokens = word_tokenize(text)
    
    # Lem word tokens and remove stop words
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    
    stems = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return stems


def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize, min_df=5)),
#     ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(min_samples_split=5 , n_estimators=25)))
    ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Uses sklearn classification report to show preciesion, recall, f1-score and support for each label
    
    input:
            model: model to test
            data: Data to test the model with
            labels: actual classification of data
            category_names: columns names of the labels
    output:
            prints classification report for the model
    """
    model_result = model.predict(X_test)

    model_result = pd.DataFrame(model_result)
    labels = pd.DataFrame(Y_test.values)

    for i in range(model_result.shape[1]):
        print('PREDICTION RESULT: {}\n\n'.format(category_names[i]), classification_report(y_true=labels[i], y_pred=model_result[i]))
        print('***' * 30, '\n')



def save_model(model, model_filepath):
    """Export model as a pickle file.
    
    input:
        model: trained model
        model_filepath: output filepath
    """
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