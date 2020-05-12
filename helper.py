import numpy as np
def calc_avg_array(transformed_matrix, doc_index_ls):
    n = len(doc_index_ls)
    unit_tfidf_shape = transformed_matrix.shape[1]
    sum_array = np.zeros(shape=(1,unit_tfidf_shape))
    for i in doc_index_ls:
        sum_array += transformed_matrix[i]
    return sum_array/n  
