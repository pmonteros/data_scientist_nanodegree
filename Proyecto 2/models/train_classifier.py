import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize

import re
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
#sklearn
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle

def load_data(database_filepath):
    """
    Loads data from database_filepath.
    
    Parameters:
    database_filepath: filepath of database
    
    Returns:
    X: message features
    y: targets
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Disaster_msg', engine)
    X = df['message']
    y = df.iloc[:, 4:]
    
    return X,y

def tokenize(text):
    """
    Function to tokenize text
    
    Parameters:
    text: text to tokenize
    
    Returns:
    lemmed: text cleaned
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    
    stemmed = [PorterStemmer().stem(w) for w in words]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed]
    
    return lemmed


def build_model():
    """
    Create a classifier model and use grid search to search best hyperparameters
    
    Returns:
    Classifier grid searched
    """
    
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
    'clf__estimator__n_estimators' : [25, 50],
    'clf__estimator__min_samples_split': [3,5]
    }

    cv = GridSearchCV(pipeline, param_grid= parameters, verbose =3)
    #cv.fit(X_train, y_train)
    return cv

def evaluate_model(model, X_test, Y_test):
    """
    Function to evaluate_model
    
    Parameters:
    model: classifier model
    X_test: test feature
    y_test: test targets
    
    Returns:
    Report for column
    """
        
    y_pred = model.predict(X_test)
    for idx, column in enumerate(y_test):
        print(column, classification_report(y_test[column], y_pred[:, idx]))


def save_model(model, model_filepath):
    """
    Save model in model_filepath
    
    Parameters:
    model: classifier model
    model_filepath: filepath to save model
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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