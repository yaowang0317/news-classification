# news-classification
Train a classification algorithm to classify news articles into a closed taxonomy system

## Workflow

**create_env.sh**: Produce environment 
**core.py**: code shared across the various stages
**fetch_dependencies.py**: Script to fetch dependencies 
**preprocess.py**: Preprocess script 
**train.py**: Train, Tune, and Evaluation
**predict.py**: Predict script that accepts a path as input (see predict_payload.json) and outputs list of lists of class probabilities. The class probabilities are placed according to the natural order defined by taxonomy_id (see taxonomy_mappings.json)
**probas.json**: output of python predict.py predict_paylaod.json
**probas.json**: output of python predict.py predict_paylaod.json


## raw_data

**rain_data.json**
a list of documents, each document will contain the following:
content.title: title of the article
content.fullTextHtml: html of the body of the article
metadata.publishedAt: datetime of publishing
content.sections: content section where the article is manually assigned into. This information will not be available anymore at prediction time due to a change in the editorial workflow.
labels: targets of the classification task. List of tuples (label, score). The labels themselves are produced by another algorithm and they contain wrongly assigned labels too. score highlights the confidence assigned by the first algorithm.

**taxonomy_mappings.json**
each key will be the integer taxonomy_id and each value will be the label.

**predict_paylaod.json**
a list of documents, each document will contain the following:
content.title
content.fullTextHtml
metadata.publishedAt

