from flask import Flask, request, jsonify

import re
import nltk
import pandas as pd
import numpy as np
import pickle
from collections import OrderedDict

from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

from pre_process_data import preprocess

app = Flask(__name__)

with open('LogisticRegression.pkl', 'rb') as handle:
    tokenizer_logistic = pickle.load(handle)

with open('vec.pkl', 'rb') as handle:
    tokenizer_tfidf = pickle.load(handle)


# request model prediction
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        a = request.args.get('query')
    if request.method == 'POST':
        a = request.form.get('query')
    
    inp_cleaned= preprocess(a)
    print(inp_cleaned)
    print(type(inp_cleaned))
    
    inp_cleaned = [inp_cleaned]
    outp_tfidf = tokenizer_tfidf.transform(inp_cleaned)
    outp_log = tokenizer_logistic.predict_proba(outp_tfidf) 
    print(outp_log)    
    percentage = np.round(outp_log*100,2)
    print(percentage)
    percentage = percentage.ravel().tolist()
    print(percentage)
    categories = ['Negative Sentiment', 'Positive Sentiment']
    output = dict(zip(categories, percentage))
    

    return output

# start Flask server

if __name__ == '__main__':
    print('Loading model...')
    
    app.run(debug=False)
