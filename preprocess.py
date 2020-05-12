from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer


def get_html_stop_words(content_body, tokenize_lemma):
    X = content_body[:10]
    stop_words_lemma = set(tokenize_lemma(' '.join(STOP_WORDS)))
    print('finish stop words in preprocess')    
    tfidf_vectorizer = TfidfVectorizer(max_features=300,
                                  stop_words=stop_words_lemma,
                                  tokenizer=tokenize_lemma)
    print('before preprocess fit')
    tfidf_vectorizer=tfidf_vectorizer.fit(X)
    print('after preprocess fit')
    html_stop_words = tfidf_vectorizer.get_feature_names()[:40]
    print('stop word in preprocess',html_stop_words)
    # stop words like /n, <p, class, http, etc.
    return html_stop_words


