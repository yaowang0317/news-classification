# news-topic-classification

Train a classification algorithm to classify news articles into a closed taxonomy system

## Workflow

**create_env.sh**

Produce environment 

**core.py**

Code shared across the various stages

**preprocess.py**

Preprocess script 

**train.py**

Train, Tune, and Evaluation.

TFIDF is chosen as the model to calculation the cosine similarity. 
One approach is to use average TFIDF. First group the document with the same label in the training dataset. Then generate a typical label matrix for each label. Next calculate the cosine similarity in the prediction dataset to each typical label. Lastly select the top 10 labels with highest cosine similarity. It can achieve almost 80% accuracy for at least one label is in the training labels, and about 40% accuracy for full coverage of the training labels. 


**predict.py**

Predict script that accepts a path as input (see raw_data/predict_payload.json) and outputs list of lists of class probabilities. The class probabilities are placed according to the natural order defined by taxonomy_id (see raw_data/taxonomy_mappings.json)

**probas.json**

Output of python predict.py predict_paylaod.json


## raw_data

**train_data.json**

a list of documents, each document will contain the following:

- *content.title*: 

title of the article

- *content.fullTextHtml*: 

html of the body of the article

- *metadata.publishedAt*: 

datetime of publishing

- *content.sections*: 

content section where the article is manually assigned into. This information will not be available anymore at prediction time due to a change in the editorial workflow.

- *labels*: 

targets of the classification task. List of tuples (label, score). The labels themselves are produced by another algorithm and they contain wrongly assigned labels too. score highlights the confidence assigned by the first algorithm.

**taxonomy_mappings.json**

each key will be the integer taxonomy_id and each value will be the label.

**predict_paylaod.json**

a list of documents, each document will contain the following:

- *content.title*

- *content.fullTextHtml*

- *metadata.publishedAt*

