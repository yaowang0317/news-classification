from core import fetch_data, read_data,write_model, tokenize_lemma, construct_typical_label_matrix, transform_matrix, get_top_10_labels
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

#construct (typical label: typical array) dictionary from the training data
def construct_label_to_doc_index_dict(train_label_dataset, index):
    label_to_doc_index_dict = defaultdict(list)
    for doc_index in range(index):
        for label in train_label_dataset[doc_index]:
            label_to_doc_index_dict[label[0]].append(doc_index)
    return label_to_doc_index_dict 


#evaluation function
#1st metrics: complete accuracy score
def evaluation_full_label_coverage(pred_labels,actual_labels):
    score_ls = []
    n = len(pred_labels)    
    for index in range(n):
        score = 0
        if actual_labels[index] != []:
            actual_label_ls = [label[0] for label in actual_labels[index]]
            for label in pred_labels[index]:
                if label[0] in actual_label_ls:
                    score += 1
            length = len(actual_labels[index])
            score_ls.append(score/length)         
    return sum(score_ls)/len(score_ls)

#2nd metrics: at least one is correct score
def evaluation_at_least_one_label_coverage(pred_labels,actual_labels):
    score_ls = []
    n = len(pred_labels)
    for index in range(n):
        score = 0
        if actual_labels[index] != []:
            actual_label_ls = [label[0] for label in actual_labels[index]]
            for label in pred_labels[index]:
                if label[0] in actual_label_ls:
                    score += 1
                    break
            score_ls.append(score)         
    return sum(score_ls)/len(score_ls)


#main function
if __name__ == "__main__":
    
    #TODO: fetch_dependencies.py 
    
    data = fetch_data("train_data.json")
    content_body = [row['content']['fullTextHtml'] for row in data]
    train_labels = [row['labels'] for row in data]
    #split training and test dataset，first 6000 docs for training, the rest is for test
    #y_train
    label_to_doc_index_dict = construct_label_to_doc_index_dict(train_labels, 6000)
    #test dataset
    y_test = train_labels[6000:]
    #modelling
    tfidf_matrix_train = transform_matrix(content_body, tokenize_lemma)
    avg_tfidf = construct_typical_label_matrix(tfidf_matrix_train, label_to_doc_index_dict)
    #save model and write out to local file 
    write_data('tfidf_model.json', avg_tfidf)
    #prediction result
    y_predict = get_top_10_labels(tfidf_matrix_train[6000:],avg_tfidf)
    #evaluation on the test dataset
    print('full label coverage accuracy score:', evaluation_full_label_coverage(y_predict,y_test))
    print('at least one label accuracy score:', evaluation_at_least_one_label_coverage(y_predict,y_test))
    
