import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import re

import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.model_selection import GridSearchCV


import pickle

def load_data(database_filepath):
    """
    load data from sqlite database

    Arguments:
        database_filepath :database path to read from

    Returns:
        X: dataframe containing features
        y: datafram containing labels

    """

    engine = create_engine('sqlite:///' + database_filepath)
    table_name = database_filepath.replace(".db", "") + "_table"
    df = pd.read_sql_table(table_name, engine)

    # Remove child alone as it has all zeros only
    df = df.drop(['child_alone'], axis=1)

    df['related'] = df['related'].map(lambda x: 1 if x == 2 else x)

    X = df['message']
    y = df.iloc[:, 4:]

    category_names = y.columns
    return X, y, category_names


def tokenize(text,url_place_holder_string="urlplaceholder"):
    """
    Tokenize the text function

    Arguments:
        text -> Text message which needs to be tokenized
    Output:
        clean_tokens -> List of tokens extracted from the provided text
    """

    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Extract all the urls from the provided text
    detected_urls = re.findall(url_regex, text)

    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)

    # Lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()

    # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return clean_tokens


# Build a custom transformer which will extract the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class

    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Given it is a tranformer we can return the self
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    """
    Build Pipeline function for ML model

    Returns:
        A Scikit ML Pipeline that process text messages and apply a classifier.

    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=50)))
    ])

    print(pipeline.get_params().keys())

    # parameters = {
    #     'features__text_pipeline__tfidf__use_idf': (True, False),
    #     'clf__estimator__n_estimators': [50, 100],
    #     'features__transformer_weights': (
    #         {'text_pipeline': 1, 'starting_verb': 0.5},
    #         {'text_pipeline': 0.5, 'starting_verb': 1},
    #     )
    # }

    # cv = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1,verbose=10)

    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
        Evaluate Model function

        This function applies a ML pipeline to a test set and prints out the model performance

        Arguments:
            pipeline : A valid scikit ML model pipline
            X_test : Test features
            Y_test : Test labels
            category_names : label names (multi-output)
    """

    Y_pred = model.predict(X_test)
    # Print the whole classification report.
    Y_pred = pd.DataFrame(Y_pred, columns=Y_test.columns)

    for column in Y_test.columns:
        print('Model Performance with Category: {}'.format(column))
        print(classification_report(Y_test[column], Y_pred[column]))


def save_model(model, model_filepath):
    """
      Save model function

      This function saves trained model as Pickle file, to be loaded later.

      Arguments:
          model -> GridSearchCV or Scikit model object
          pickle_filepath -> destination path to save .pkl file

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

        # print("\nBest Parameters:", model.best_params_)

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