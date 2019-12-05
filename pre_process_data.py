"""
Author: Saurabh Annadate

This script contains all functions that help with text analytics
"""

import logging
import os
import pandas as pd
import logging


import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

import re 

logger = logging.getLogger()


stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")
decode_map = {0: -1, 2: 0, 4: 1}
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

def label_sentiment(label):
    return decode_map[int(label)]


def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

#data.target = data.target.apply(lambda x: label_sentiment(x))
#data.text = data.text.apply(lambda x: preprocess(x))
    


