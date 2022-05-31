
#importing libraries
import os
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import string
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models import word2vec
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords #for importing stopwords
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import gensim
import time
from nltk.stem import PorterStemmer
import copy
from copy import deepcopy
import sys
from gensim.models import word2vec


# Set values for parameters of word_to_vec
num_features = 600    # Word vector dimensionality
min_word_count = 15   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 20          # Context window size
downsampling = 1e-5   # Downsample setting for frequent words

#Parameters for HD and Random Projection
max_score = -np.inf
max_index = -1
D = 5000
log = False
epoch=100
alpha=1.0

train_test_ratio = 0.3

#libraries
def process_tweet(review):
    review = review.replace("<body>","")
    review = review.replace("</body>","")
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

def encoding_rp(X_data, base_matrix, quantize=False):
    enc_hv = []
    issue = []
    issue_len = []
    for i in range(len(X_data)):
        if i % 1000 == 0:
            if log:
                sys.stdout.write(str(int(i/len(X_data)*100)) + '% ')
                sys.stdout.flush()
            print(np.round(i/len(X_data)*100,2), " % ...")
        if len(X_data[i]) != 600:
            issue_len.append(len(X_data[i]))
            X_data[i] = X_data[i][:600]
            issue.append(i)
        hv = np.matmul(base_matrix, X_data[i])
        if quantize:
            hv = binarize(hv)
        enc_hv.append(hv)
    return enc_hv, issue, issue_len

def max_match_(class_hvs, enc_hv, class_norms):
        max_score = -np.inf
        max_index = -1
        for i in range(len(class_hvs)):
        #print("***",len(class_hvs[i]), len(enc_hv))
            score = np.matmul(class_hvs[i], enc_hv) / class_norms[i]
            if score > max_score:
                max_score = score
                max_index = i
        return max_index

/ class_norms[i]
epoch = 1


def load_data(path):
    messy_docs = []
    docs = []
    filenames = []
    for f in os.listdir(path):
        d = []
        if f.endswith('.sgm'):
            filenames.append(f)
            file = open(path+f, 'r', encoding='utf-8', errors='ignore')
            file = file.read()
            messy_docs.append(file)

            #passing to BeautifulSoup for parsing the semi-structured text with tags
            soup = BeautifulSoup(file, 'html.parser')
            contents = soup.findAll('body')
            for c in contents:
                d.append(c)
            docs.append(d)
    print('There exist {} docs.'.format(len(docs)))
    return docs

def plot_reuters_counts(docs):
    counts = []
    for i in range(len(docs)):
        counts.append(len(docs[i]))
    plt.figure(figsize=(8,4))
    plt.xticks(rotation='vertical')
    plt.bar(filenames, counts, color="purple")
    plt.title("Number of contents in each doc.")
    plt.savefig("corpus_counts.png")

def unique_words(X):
    unique_ws = {}
    for content in X:
        for word in content:
            if word not in unique_words:
                unique_ws[word] = 1
            else:
                unique_ws[word] += 1
    return unique_ws

def filter_uncommon_words(unique_ws):
    decreased_size_dict = {}
    for k,v in unique_ws.items():
        if v >5 and len(k) >2:
            decreased_size_dict[k] = v
    return decreased_size_dict

def w2v_model_create(X_train, num_workers=600, num_features=15, min_word_count=4, context=20, downsampling=1e-5):
    print ("Training model...")
    w2v_model = word2vec.Word2Vec(X_train, workers=num_workers, vector_size=num_features, min_count = min_word_count, window = context, sample = downsampling)
    w2v_model.init_sims(replace=True)
    model_name = "reuters_300features_40minwords_10context"
    w2v_model.save(model_name)
    return w2v_model

def encode_sentences_w2v(X_train, w2v_D, w2v_model):
    s = time.time()
    train_embedding = []
    for i in range(len(X_train)):
        if i %1000 == 0:
            print(np.round(i/len(X_train)*100,2), " % ...")
        encoded_sentences_vec = np.zeros(w2v_D)
        for w in X_train[i]:
            vec = np.zeros(w2v_D)
            try:
                vec = w2v_model.wv[w]
            except:
                pass
            encoded_sentences_vec =  [j+k for j,k in zip(vec, encoded_sentences_vec)]
        train_embedding.append(encoded_sentences_vec)
    print("creating train embedding took {} minutes".format((time.time()-s)/60))
    return train_embedding

def test_evaluation(encoded_test, y_test, D):
    class_hvs = [[0.] * D] * len(np.unique(y_test))
    indices = {}
    for i in list(np.unique(y_test)):
        indices[i] = []
    for i in range(0, len(encoded_train[0])):
        for j in list(np.unique(y_test)):
            if y_train[i]==j:
                indices[j].append(i)
                class_hvs[j] = [m+n for m,n in zip(encoded_train[0][i], class_hvs[j])]
        if i%5000==0:
            print(np.round(i/len(encoded_train[0])*100,2), " % ...")

    class_norms = [np.linalg.norm(hv) for hv in class_hvs]
    class_hvs_best = deepcopy(class_hvs)
    class_norms_best = deepcopy(class_norms)
    if log: print('\n\nEncoding ' + str(len(X_test)) + ' test data')
    correct = 0
    preds = []
    for i in range(len(encoded_test[0])):
        if i % 500==0:
            print(np.round(encoded_test[0]*100), " % ...")
        predict = max_match(class_hvs, encoded_test[0][-1], class_norms)
        #predict = max_match(class_hvs_best, test_enc_hvs[i], class_norms_best)
        if predict == y_test[i]:
            correct += 1
        preds.append(predict)
    acc = float(correct)/len(encoded_test[0])
    #print(time.time() - start)
    print("accuracy is {}".format(acc))
    return acc, preds



#reading Reuters-21578
path = '/home/eagle/fatemeh/reuters21578/'
docs = load_data(path)
X = []
Y = []
for i in range(len(docs)):
    print("Loading doc {} ... ".format(i))
    for j in range(len(docs[i])):
        processed = process_tweet(str(docs[i][j]))
        X.append(processed)
        Y.append(i)
#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=train_test_ratio, random_state=42)
#word-to-vec model creation
m = w2v_model_create(X_train, num_workers, num_features, min_word_count, context, downsampling)
#train and test sentences encoding
train_embedding = encode_sentences_w2v(X_train, 600, m)
test_embedding = encode_sentences_w2v(X_test, 600, m)

#encode to hypervectors using random projection
base_matrix = np.random.rand(D, len(train_embedding[0]))
base_matrix = np.where(base_matrix > 0.5, 1, -1)
base_matrix = np.array(base_matrix, np.int8)
encoded_train = encoding_rp(train_embedding, base_matrix, quantize=False)
encoded_test = encoding_rp(test_embedding, base_matrix, quantize=False)

encoded_test = encoded_test[0]
encoded_train = encoded_train[0]
print("Accuracy is {}".format(test_evaluation[0]))
