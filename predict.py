from sklearn.feature_extraction.text import TfidfVectorizer
from core import fetch_data, read_model, write_data, construct_typical_label_matrix, transform_matrix, tokenize_lemma, get_top_10_labels
import sys

def predict(X_pred, tokenize_lemma, typical_label_matrix):
    tfidf_matrix_pred = transform_matrix(X_pred, tokenize_lemma)
    output = get_top_10_labels(tfidf_matrix_pred, typical_label_matrix)
    return output

if __name__ == "__main__":
   
    predict_payload = fetch_data("raw_data/predict_payload.json")
    content_body = [row['content']['fullTextHtml'] for row in predict_payload]
    avg_tfidf = read_model("tfidf_model.json")
    output = predict(content_body, tokenize_lemma, avg_tfidf)
    write_data("probas.json", output)
    
