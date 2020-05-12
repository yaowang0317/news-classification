import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import base
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.model_selection import train_test_split
from collections import defaultdict
import ujson as json 
import pickle
import sys
