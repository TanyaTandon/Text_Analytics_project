import logging
import os
import pandas as pd
import yaml
import logging
import datetime
import pickle

from pre_process_data import label_sentiment
from pre_process_data import preprocess

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, f1_score, recall_score, precision_score

logger = logging.getLogger()

FILTER_STEM = True
TRAIN_SIZE = 0.8
RANDOM_STATE = 7
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"

def fit_logistic_regression():
    """
    Fit a logistic regression on the data to model the review

    Args:
        None
    
    Returns:
        None
    """

    logger.debug("Running the fit_logistic_regression function now")


    #Loading and pre processing the data
    dataset_filename = "training.1600000.processed.noemoticon.csv"
    dataset_path = os.path.join("data",dataset_filename)
    print("Open file:", dataset_path)
    data = pd.read_csv(dataset_path, encoding =DATASET_ENCODING , names=DATASET_COLUMNS)
    
    data.target = data.target.apply(lambda x: label_sentiment(x))
    data.text = data.text.apply(lambda x: preprocess(x))

    vectorizer = TfidfVectorizer()
    word_frequency = vectorizer.fit_transform(data.text)
    sample_index = np.random.random(data.shape[0])
    X_train, X_test = word_frequency[sample_index <= TRAIN_SIZE, :], word_frequency[sample_index > TRAIN_SIZE, :]
    Y_train, Y_test = data.target[sample_index <= TRAIN_SIZE], df.target[sample_index > TRAIN_SIZE]

    model = LogisticRegression(random_state=0, solver='saga', multi_class='multinomial').fit(X_train, Y_train)

    pickle.dump(model, open("LogisticRegression.pkl", "wb"))

    return 



