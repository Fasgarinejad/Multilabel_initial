#importing the libraries, stopwords, etc.
import nltk
nltk.download('stopwords')
#from utils import process_tweet, build_freqs
# importing the libraries
import re #for regular expression operations
import string #for string operations
from nltk.corpus import stopwords #for importing stopwords
from nltk.stem import PorterStemmer #for importing stemmer
from nltk.tokenize import TweetTokenizer #for importing tweet tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from itertools import islice #iter in dictionary
from keras.preprocessing import image #for plotting the wordcloud
import seaborn as sns
import gc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc
from nltk.tokenize import word_tokenize
from scipy import sparse
from nltk.corpus import twitter_samples

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

import numpy as np
import pickle
import sys
import time
from copy import deepcopy

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def process_tweet(review):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    review_tokens = review.split()
    cleaned_review = []
    for word in review_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            stem_word = stemmer.stem(word)  # stemming word
            cleaned_review.append(stem_word)
    return cleaned_review

def max_match(class_hvs, enc_hv, class_norms):
        max_score = -np.inf
        max_index = -1
        for i in range(len(class_hvs)):
        #print("***",len(class_hvs[i]), len(enc_hv))
            score = np.matmul(class_hvs[i], enc_hv) / class_norms[i]
            if score > max_score:
                max_score = score
                max_index = i
        return max_index

#importing the training and test data from NLTK's twitter_samples
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg
# combinung positive and negative labels
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

cleaned_review = []
for sentence in train_x:
    cleaned_review.append(process_tweet(sentence))

cleaned_test = []
for sentence in test_x:
    cleaned_test.append(process_tweet(sentence))

sentences = cleaned_review

unique_words = {}
for i in range(len(cleaned_review)):
    for j in range(len(cleaned_review[i])):
        word = cleaned_review[i][j]
        if word not in unique_words:
            unique_words[word] = 1
        else:
            unique_words[word] += 1

# Set values for various parameters
num_features = 535    # Word vector dimensionality
min_word_count = 15   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 20          # Context window size
downsampling = 1e-5   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print ("Training model...")
w2v_model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
w2v_model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
w2v_model.save(model_name)

v = []
for i in range(len(sentences)):
    encoded_sentences_vec = np.zeros(535) #535
    for w in sentences[i]:
        vec = np.zeros(535)
        try:
            vec = w2v_model[w]
        except:
            pass
        encoded_sentences_vec =  [j+k for j,k in zip(vec, encoded_sentences_vec)]
    v.append(encoded_sentences_vec)

v_string = ''
for i in range(len(v)):
    v_string += str(v[i]) + "\n"
with open("train_x.txt", "w") as output:
    output.write(str(v_string))

class_hvs = [[0.] * 535] * (2)
for i in range(0, len(v)):
    if train_y[i] == 1:
        class_hvs[1] = [m+n for m,n in zip(v[i],class_hvs[1])] #positive
    else:
        class_hvs[0] = [m+n for m,n in zip(v[i],class_hvs[0])] #negative


test_v = []
for i in range(len(test_x)):
    test_encoded_sentences_vec = np.zeros(535)
    for w in cleaned_test[i]:
        vec = np.zeros(535)
        try:
            vec = w2v_model[w]
        except:
            pass
        test_encoded_sentences_vec =  [j+k for j,k in zip(vec, test_encoded_sentences_vec)]
    test_v.append(test_encoded_sentences_vec)

def encoding_rp(X_data, base_matrix, quantize=False):
	enc_hv = []
	issue = []
	issue_len = []
	for i in range(len(X_data)):
		#print(i, base_matrix.shape, len(X_data[i]))
		if len(X_data[i]) != 535:
			issue_len.append(len(X_data[i]))
			X_data[i] = X_data[i][:535]
			issue.append(i)
		hv = np.matmul(base_matrix, X_data[i])
		if quantize:
			hv = binarize(hv)
		enc_hv.append(hv)
	return enc_hv, issue, issue_len
X_train = v
y_train = train_y
X_test = test_v
y_test = test_y
D = 5000
base_matrix = np.random.rand(D, len(v[0]))
base_matrix = np.where(base_matrix > 0.5, 1, -1)
base_matrix = np.array(base_matrix, np.int8)
# class_hvs = [[0.] * D] * (max(y_train) + 1)
class_hvs = [[0.] * D] * 2
#train_enc_hvs = encoding_rp(X_train, base_matrix, quantize=False)

X_train = v
y_train = train_y
X_test = test_v
y_test = test_y
D = 5000
base_matrix = np.random.rand(D, len(v[0]))
base_matrix = np.where(base_matrix > 0.5, 1, -1)
base_matrix = np.array(base_matrix, np.int8)
# class_hvs = [[0.] * D] * (max(y_train) + 1)
class_hvs = [[0.] * D] * 2
#train_enc_hvs = encoding_rp(X_train, base_matrix, quantize=False)

train_enc_hvs, issue, issue_len = encoding_rp(v, base_matrix, quantize=False)

alg='rp'
epoch=100
alpha=1.0
log=True
for i in range(len(train_enc_hvs)):
    if i%1000 == 0:
        print(np.round(i/len(train_enc_hvs)*100, 2))
    class_hvs[int(y_train[i].astype(int))] += train_enc_hvs[i]
    #class_hvs[y_train[i]] = [j+k for j,k in zip(train_enc_hvs[i], class_hvs[y_train[i]])]
class_norms = [np.linalg.norm(hv) for hv in class_hvs]
class_hvs_best = deepcopy(class_hvs)
class_norms_best = deepcopy(class_norms)

validation_enc_hvs = encoding_rp(test_v, base_matrix, quantize=False)
t = [] #important
for i in range(2000):
    t.append(validation_enc_hvs[0][i])

epoch = 100
if epoch > 0:
    acc_max = -np.inf
    if log: print('\n\n' + str(epoch) + ' retraining epochs')
    for i in range(epoch):
        for j in range(len(train_enc_hvs)):
            #print(j)
            predict = max_match(class_hvs, train_enc_hvs[j], class_norms)
            if predict != y_train[j]:
                class_hvs[predict] -= np.multiply(alpha/(1. + epoch/5.), train_enc_hvs[j])
                class_hvs[int(y_train[i].astype(int))] += np.multiply(alpha/(1. + epoch/5.), train_enc_hvs[j])
        class_norms = [np.linalg.norm(hv) for hv in class_hvs]
        correct = 0
        for j in range(len(t)):
            predict = max_match(class_hvs, t[j], class_norms)
            if predict == test_y[j]:
                correct += 1
        acc = float(correct)/len(validation_enc_hvs)
        if log:
            sys.stdout.write("%.4f " %acc)
            sys.stdout.flush()
        if acc > acc_max:
            acc_max = acc
            class_hvs_best = deepcopy(class_hvs)
            class_norms_best = deepcopy(class_norms)
start = time.time()
if log: print('\n\nEncoding ' + str(len(X_test)) + ' test data')
test_enc_hvs = encoding_rp(X_test, base_matrix, quantize=False)
correct = 0

preds = []
correct = 0
for i in range(len(t)):
    predict = max_match(class_hvs_best, t[i], class_norms_best)
    if predict == y_test[i]:
        correct += 1
    preds.append(predict)
acc = float(correct)/len(t)
#print(time.time() - start)
#2200 features, D=5k
print("test evluation")
acc = accuracy_score(preds, test_y)
recall = recall_score(preds, test_y)
precision  = precision_score(preds, test_y)
print("acc: {} | recall: {} | precision: {}".format(acc, recall, precision))
