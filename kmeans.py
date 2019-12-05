import gensim
from gensim.models.phrases import Phrases, Phraser
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from IPython.display import display

from pre_process_data import label_sentiment
from pre_process_data import preprocess


FILTER_STEM = True
TRAIN_SIZE = 0.8
RANDOM_STATE = 7
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"

def k_means_sentiment():

	dataset_filename = "training.1600000.processed.noemoticon.csv"
    dataset_path = os.path.join("data",dataset_filename)
    print("Open file:", dataset_path)
    data = pd.read_csv(dataset_path, encoding =DATASET_ENCODING , names=DATASET_COLUMNS)
    
    data.target = data.target.apply(lambda x: label_sentiment(x))
    data.text = data.text.apply(lambda x: preprocess(x))


	sent = [row for row in data.text]
	phrases = Phrases(sent, min_count=1, progress_per=50000)
	bigram = Phraser(phrases)
	sentences = bigram[sent]



	w2v_model = gensim.models.word2vec.Word2Vec(min_count=3,
	                     window=4,
	                     size=300,
	                     sample=1e-5, 
	                     alpha=0.03, 
	                     min_alpha=0.0007, 
	                     negative=20,
	                     workers=8 )



	w2v_model.build_vocab(sentences, progress_per=50000)

	w2v_model.save("word2vec.model")

	word_vectors = Word2Vec.load("word2vec.model").wv

	model = KMeans(n_clusters=2, max_iter=1000, random_state=True, n_init=50).fit(X=word_vectors.vectors)

	word_vectors.similar_by_vector(model.cluster_centers_[0], topn=10, restrict_vocab=None)

	positive_cluster_center = model.cluster_centers_[0]
	negative_cluster_center = model.cluster_centers_[1]

	words = pd.DataFrame(word_vectors.vocab.keys())
	words.columns = ['words']
	words['vectors'] = words.words.apply(lambda x: word_vectors.wv[f'{x}'])
	words['cluster'] = words.vectors.apply(lambda x: model.predict([np.array(x)]))
	words.cluster = words.cluster.apply(lambda x: x[0])
	words['cluster_value'] = [1 if i==0 else -1 for i in words.cluster]
	words['closeness_score'] = words.apply(lambda x: 1/(model.transform([x.vectors]).min()), axis=1)
	words['sentiment_coeff'] = words.closeness_score * words.cluster_value
	words[['words', 'sentiment_coeff']].to_csv('sentiment_dictionary.csv', index=False)



	file_model = data.copy()
	file_model = file_model[data.text.str.len()>1]
	file_export = file_model.copy()
	file_export['old_title'] = file_export.text
	file_export.old_title = file_export.old_title.str.join(' ')
	file_export.text = file_export.text.apply(lambda x: ' '.join(bigram[x]))
	file_export.target = file_export.target.astype('int8')
	file_export[['text', 'target']].to_csv('cleaned_dataset.csv', index=False)
	sentiment_map = pd.read_csv('sentiment_dictionary.csv')
	sentiment_dict = dict(zip(sentiment_map.words.values, sentiment_map.sentiment_coeff.values))
	final_file = pd.read_csv('cleaned_dataset.csv')
	file_weighting = final_file.copy()

	tfidf = TfidfVectorizer(tokenizer=lambda y: y.split(), norm=None)
	tfidf.fit(file_weighting.text)
	features = pd.Series(tfidf.get_feature_names())
	transformed = tfidf.transform(file_weighting.text)
	replaced_tfidf_scores = file_weighting.apply(lambda x: replace_tfidf_words(x, transformed, features), axis=1)
	replaced_closeness_scores = file_weighting.text.apply(lambda x: list(map(lambda y: replace_sentiment_words(y, sentiment_dict), x.split())))
	replacement_df = pd.DataFrame(data=[replaced_closeness_scores, replaced_tfidf_scores, file_weighting.text, file_weighting.target]).T
	replacement_df.columns = ['sentiment_coeff', 'tfidf_scores', 'sentence', 'sentiment']
	replacement_df['sentiment_rate'] = replacement_df.apply(lambda x: np.array(x.loc['sentiment_coeff']) @ np.array(x.loc['tfidf_scores']), axis=1)
	replacement_df['prediction'] = (replacement_df.sentiment_rate>0).astype('int8')
	replacement_df['sentiment'] = [1 if i==1 else 0 for i in replacement_df.sentiment]


	predicted_classes = replacement_df.prediction
	y_test = replacement_df.sentiment

	conf_matrix = pd.DataFrame(confusion_matrix(replacement_df.sentiment, replacement_df.prediction))
	print('Confusion Matrix')
	display(conf_matrix)

	test_scores = accuracy_score(y_test,predicted_classes), precision_score(y_test, predicted_classes), recall_score(y_test, predicted_classes), f1_score(y_test, predicted_classes)

	print('\n \n Scores')
	scores = pd.DataFrame(data=[test_scores])
	scores.columns = ['accuracy', 'precision', 'recall', 'f1']
	scores = scores.T
	scores.columns = ['scores']
	display(scores)


def create_tfidf_dictionary(x, transformed_file, features):
    '''
    create dictionary for each input sentence x, where each word has assigned its tfidf score
    
    x - row of dataframe, containing sentences, and their indexes,
    transformed_file - all sentences transformed with TfidfVectorizer
    features - names of all words in corpus used in TfidfVectorizer

    '''
    vector_coo = transformed_file[x.name].tocoo()
    vector_coo.col = features.iloc[vector_coo.col].values
    dict_from_coo = dict(zip(vector_coo.col, vector_coo.data))
    return dict_from_coo

def replace_tfidf_words(x, transformed_file, features):
    '''
    replacing each word with it's calculated tfidf dictionary with scores of each word
    x - row of dataframe, containing sentences, and their indexes,
    transformed_file - all sentences transformed with TfidfVectorizer
    features - names of all words in corpus used in TfidfVectorizer
    '''
    dictionary = create_tfidf_dictionary(x, transformed_file, features)   
    return list(map(lambda y:dictionary[f'{y}'], x.text.split()))












