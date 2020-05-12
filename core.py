#ujson has the same interface as the built-in json library, but is substantially faster 
#(at the cost of non-robust handling of malformed json)
import ujson as json
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from helper import calc_avg_array 
from preprocess import get_html_stop_words
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def fetch_data(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f][0] #organize training data inside a list comprehension to get a list of dictionaries
    return data

def write_data(file_path, data):
    with open(file_path, "wb") as f:
        pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)

def read_model(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f) 
    return data

def tokenize_lemma(text):
    nlp = spacy.load("en_core_web_sm")
    return [w.lemma_.lower() for w in nlp(text)]

def construct_typical_label_matrix(transformed_matrix, label_to_doc_index_dict):
    label_to_typical_matrix_dict= defaultdict()
    for label,doc_index_ls in label_to_doc_index_dict.items():
        label_to_typical_matrix_dict[label] = calc_avg_array(transformed_matrix,doc_index_ls)
    return label_to_typical_matrix_dict

def transform_matrix(content_body, tokenize_lemma):
    html_stop_words = get_html_stop_words(content_body,tokenize_lemma)
    stop_words_lemma_train = set(tokenize_lemma(' '.join(STOP_WORDS.union(set(html_stop_words)))))
    X = content_body
    tfidf_vectorizer = TfidfVectorizer(max_features=300,
                                  stop_words=stop_words_lemma_train,
                                  tokenizer=tokenize_lemma)
    tfidf_vectorizer=tfidf_vectorizer.fit(X)
    tfidf_matrix = tfidf_vectorizer.transform(X)
    return tfidf_matrix

#prediction function to get top 10 labels with highest cosine similarity score
def get_top_10_labels(transformed_matrix, typical_label_matrix):
    result_ls = []
    for array in range(transformed_matrix.shape[0]):
        ls = []
        for typical_label,typical_matrix in typical_label_matrix.items():
            ls.append([typical_label,cosine_similarity(transformed_matrix[array], typical_matrix)])
            ls.sort(key = lambda x:x[1],reverse = True)
        result = ls[:10]
        result.sort(key=lambda v:v[0])
        result_ls.append(result)
    return result_ls
