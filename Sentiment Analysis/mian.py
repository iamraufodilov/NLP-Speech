# import nexessary libraries
import re
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report


# load the dataset
DATASET_COLUMNS=['target','ids','date','flag','user','text']
DATASET_ENCODING = "ISO-8859-1"
data_path = "G:/rauf/STEPBYSTEP/Data2/twitter 1.6m/training.1600000.processed.noemoticon.csv"
data = pd.read_csv(data_path, encoding = DATASET_ENCODING, names = DATASET_COLUMNS)
print(data.head(5))


# data preprocessing
data = data[['text','target']]
print(data.head(5))

# assign 1 for posetive sentence 4
data['target'] = data['target'].replace(4, 1)

# separating posetive and negative sentences
data_pos = data[data['target'] == 1]
data_neg = data[data['target'] == 0]

data_pos = data_pos.iloc[:int(20000)]
data_neg = data_neg.iloc[:int(20000)]

dataset = pd.concat([data_pos, data_neg])

dataset['text'] = dataset['text'].str.lower()
dataset['text'].tail()
print(dataset.head(5))

# cleaning text from stopwords
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']

STOPWORDS = set(stopwordlist)
def clean_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

dataset['text'] = dataset['text'].apply(lambda text: clean_stopwords(text))
print(dataset.head(5))

# cleaning and removing punctuations
english_punctuations = string.punctuation
def cleaning_punctuations(text):
    translator = str.maketrans('','', english_punctuations)
    return text.translate(translator)

dataset['text'] = dataset['text'].apply(lambda text: cleaning_punctuations(text))
print(dataset.head(5))

# very nice keep going

# clean repeating characters 
def clean_rep_char(text):
    return re.sub(r'(.)1+', r'1', text)
dataset['text'] = dataset['text'].apply(lambda text: clean_rep_char(text))
print(dataset.head(5))


# cleaning and removing urls
def cleaning_urls(text):
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',text)
dataset['text'] = dataset['text'].apply(lambda text: cleaning_urls(text))
print(dataset.head(5))


# cleaning numbers from text
def clean_numbers(text):
    return re.sub('[0-9]+', '', text)
dataset['text'] = dataset['text'].apply(lambda text: clean_numbers(text))
print(dataset.head(5))


# tokenize tweets
tokenizer = RegexpTokenizer(r'w+')
dataset['text'] = dataset['text'].apply(tokenizer.tokenize)
dataset['text'].head()


# Stemming
def stemming(data):
    stemmer = nltk.PorterStemmer()
    text = [stemmer.stem(word) for word in data]
    return text
dataset['text'] = dataset['text'].apply(lambda text: stemming(text))
print(dataset.head(5))


# Lemmatizing
lm = nltk.stem.WordNetLemmatizer()
def text_lemmatizing(data):
    text = [lm.lemmatize(word) for word in data]
    return text
dataset['text'] = dataset['text'].apply(lambda text: text_lemmatizing(text))
print(dataset.head(5))


# separting feature and label
X = dataset['text']
y = dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)


# transforming dataset using tfidfvectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)


# model evaluation
def model_evaluate(model, X_test, y_pred):
    y_pred = model.predict(X_test)
    #print evaluation metrics
    print(classification_report(y_test, y_pred))
    # plot confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    print(cf_matrix)


# model building
# here we can choose lots of classification models 
# but now I choose Logistic Regression
lrmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
lrmodel.fit(X_train, X_test)
print(model_evaluate(lrmodel, X_test, y_test))
