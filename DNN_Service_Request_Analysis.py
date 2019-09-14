######SR keywords Classification to Root cause using DNN######
import pandas as pd
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import string
import unicodedata
import sys
from ast import literal_eval
import pickle
import random

def unique_words(sentence, number):
    return [w for w in set(sentence.translate(None, punctuation).lower().split()) if len(w) >= number]

def preprocessing(line):
    line = line.lower()
    line = re.sub(r"[{}]".format(string.punctuation), " ", line)
    return line

def remove_punctuation(text):
    return text.translate(tbl)

tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith('P'))

#df = pd.read_csv("All_SRNotes_Extracted.csv", encoding='utf-8')
df = pd.read_csv('cpn_sr_asr9k.csv', encoding = 'utf-8')
df = df.dropna(subset=['SR_Keywords'])
df = df[df['Year'] == 2018]

counts = df['ROOT_CAUSE'].value_counts()
df = df[~df['ROOT_CAUSE'].isin(counts[counts < 3].index)]


msk = np.random.rand(len(df)) < 0.8
df_old = df[~msk]
df = df[msk]

a = list(df['ROOT_CAUSE'].unique())
b = list(df_old['ROOT_CAUSE'].unique())
def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 

print(len(a))
print(len(b))
print(len(intersection(a, b)))

#df['SR_Keywords'] = df['SR_Keywords'].fillna('')
#If groupby not done...use below code
#df = df.groupby('sr_number').agg({'sr_number':'first', 'Keywords': ', '.join})
#df = df.groupby('SR_NUMBER').agg({'SR_NUMBER':'first', 'Keywords_SR': ', '.join, 'FA Label':'first'})
#old_df = df.copy()
######Mark any label less than certain value as OTHERS######
#b = df['FA Label'].value_counts()[0]/10
#a = df['FA Label'].value_counts()
#m = df['FA Label'].isin(a.index[a < b])
#df.loc[m, 'FA Label'] = 'OTHER'


fa_label = df['ROOT_CAUSE'].tolist()
keywords = df['SR_Keywords'].values.tolist()
for i in range(len(keywords)):
    keywords[i] = keywords[i].replace('[], ', '')
    keywords[i] = keywords[i].replace('], [', ',')

docs = []
for i in range(len(keywords)):
    s = ''
    if(len(keywords[i].replace(',', '').replace(' ', '')) > 2):
        for keyword in literal_eval(keywords[i])[:]:
            #print(keyword)
            keyword = keyword.split(':')[0]
            #s = s + (' '.join(set(keyword.split()))) + ','
            s = s + keyword + ','
            #print(s)
        docs.append(s.rstrip(','))
    else:
        docs.append(' ')

d = pd.DataFrame()
d['keywords'] = docs
d['Label'] = fa_label

d = d.groupby('Label').agg({'Label':'first', 'keywords': ', '.join})
unique_labels = d['Label'].tolist()

training_data = {}
for i in range(d.shape[0]):
    training_data[d.iloc[i]['Label']] = d.iloc[i]['keywords'].split(',')


data = training_data

# get a list of all categories to train for
categories = list(data.keys())
#Write this to text file
with open('Categories_cpn.txt', 'wb') as f:
    pickle.dump(categories, f)

words = []
# a list of tuples with words in the sentence and category name
docs = []

for each_category in data.keys():
    for each_sentence in data[each_category]:
        # remove any punctuation from the sentence
        each_sentence = remove_punctuation(each_sentence)
        # extract words from each sentence and append to the word list
        w = nltk.word_tokenize(each_sentence)
        words.extend(w)
        docs.append((w, each_category))


# stem and lower each word and remove duplicates
stemmer = LancasterStemmer()
words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))

with open('Words_cpn.txt', 'wb') as f:
    pickle.dump(words, f)


#print(words)
#print(docs)

# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(categories)

for doc in docs:
    if(doc[0] != []):
        # initialize our bag of words(bow) for each document in the list
        bow = []
        # list of tokenized words for the pattern
        token_words = doc[0]
        # stem each word
        token_words = [stemmer.stem(word.lower()) for word in token_words]
        # create our bag of words array
        for w in words:
            bow.append(1) if w in token_words else bow.append(0)
        output_row = list(output_empty)
        output_row[categories.index(doc[1])] = 1
        # our training set will contain a the bag of words model and the output row that tells
        # which catefory that bow belongs to.
        training.append([bow, output_row])

# shuffle our features and turn into np.array as tensorflow  takes in numpy array
random.shuffle(training)
training = np.array(training)
 
# trainX contains the Bag of words and train_y contains the label/ category
train_x = list(training[:,0])
train_y = list(training[:,1])

# reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=100, batch_size=8, show_metric=True)
model.save('fa_model_cpn.tflearn')


######Testing on 2018 data######
#df = pd.read_csv('cpn_sr_asr9k.csv', encoding = 'utf-8')
#df = df.dropna(subset=['SR_Keywords'])
#df = df[df['Year'] == 2018]
df = df_old

fa_label = df['ROOT_CAUSE'].tolist()
keywords = df['SR_Keywords'].values.tolist()
for i in range(len(keywords)):
    keywords[i] = keywords[i].replace('[], ', '')
    keywords[i] = keywords[i].replace('], [', ',')

docs = []
for i in range(len(keywords)):
    s = ''
    if(len(keywords[i].replace(',', '').replace(' ', '')) > 2):
        for keyword in literal_eval(keywords[i])[:]:
            #print(keyword)
            keyword = keyword.split(':')[0]
            #s = s + (' '.join(set(keyword.split()))) + ','
            s = s + keyword + ','
            #print(s)
        docs.append(s.rstrip(','))
    else:
        docs.append(' ')

with open('Words_cpn.txt', 'rb') as f:
    words = pickle.load(f)

with open('Categories_cpn.txt', 'rb') as f:
    categories = pickle.load(f)

new_docs = []
for each_sentence in docs:
    # remove any punctuation from the sentence
    each_sentence = remove_punctuation(each_sentence.replace(',', ', '))
    # extract words from each sentence and append to the word list
    w = nltk.word_tokenize(each_sentence)
    #print ("tokenized words: ", w)
    new_docs.append(' '.join(w))


tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model2 = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
#model.fit(train_x, train_y, n_epoch=100, batch_size=8, show_metric=True)
model2.load('fa_model_cpn.tflearn')

def get_tf_record(sentence):
    global words
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    # bag of words
    bow = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1
    return(np.array(bow))

categories_pred = []
for doc in new_docs:
    a = np.argsort(model2.predict([get_tf_record(doc)])[0])[-5:]
    pred = []
    for i in a:
        pred.append(categories[i])
    categories_pred.append(pred)

c = 0
true_predictions = []
for i in range(len(fa_label)):
    if(fa_label[i] in categories_pred[i]):
        c = c + 1
        true_predictions.append(1)
    else:
        true_predictions.append(0)

print('Accuracy: ', float(c/len(fa_label)))
df['CPN_Prediction'] = categories_pred
df['True_Predictions'] = true_predictions
df.to_csv('CPN_Predictions_2018.csv', index = False, encoding = 'utf-8')


k = 0
for i in list(set(fa_label)):
    if(i in categories):
        k = k + 1

print(float(k/len(list(set(fa_label)))))







#########USING FA_CDETS#########
######SR Classification using DNN######
import pandas as pd
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import string
import unicodedata
import sys
from ast import literal_eval
import pickle

def unique_words(sentence, number):
    return [w for w in set(sentence.translate(None, punctuation).lower().split()) if len(w) >= number]

def preprocessing(line):
    line = line.lower()
    line = re.sub(r"[{}]".format(string.punctuation), " ", line)
    return line

def remove_punctuation(text):
    return text.translate(tbl)


tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith('P'))

#df = pd.read_csv("All_SRNotes_Extracted.csv", encoding='utf-8')
df = pd.read_csv('pid_sr_asr9k.csv', encoding = 'utf-8')
df = df.dropna(subset=['SR_Keywords'])
df = df.dropna(subset=['FA_CDETS'])
df = df[df['Year'] == 2017]
#df['SR_Keywords'] = df['SR_Keywords'].fillna('')
#If groupby not done...use below code
#df = df.groupby('sr_number').agg({'sr_number':'first', 'Keywords': ', '.join})
#df = df.groupby('SR_NUMBER').agg({'SR_NUMBER':'first', 'Keywords_SR': ', '.join, 'FA Label':'first'})
#old_df = df.copy()
######Mark any label less than certain value as OTHERS######
#b = df['FA Label'].value_counts()[0]/10
#a = df['FA Label'].value_counts()
#m = df['FA Label'].isin(a.index[a < b])
#df.loc[m, 'FA Label'] = 'OTHER'


fa_label = df['FA_CDETS'].tolist()
keywords = df['SR_Keywords'].values.tolist()
for i in range(len(keywords)):
    keywords[i] = keywords[i].replace('[], ', '')
    keywords[i] = keywords[i].replace('], [', ',')

docs = []
for i in range(len(keywords)):
    s = ''
    if(len(keywords[i].replace(',', '').replace(' ', '')) > 2):
        for keyword in literal_eval(keywords[i])[:]:
            #print(keyword)
            keyword = keyword.split(':')[0]
            #s = s + (' '.join(set(keyword.split()))) + ','
            s = s + keyword + ','
            #print(s)
        docs.append(s.rstrip(','))
    else:
        docs.append(' ')

d = pd.DataFrame()
d['keywords'] = docs
d['Label'] = fa_label

d = d.groupby('Label').agg({'Label':'first', 'keywords': ', '.join})
unique_labels = d['Label'].tolist()

training_data = {}
for i in range(d.shape[0]):
    training_data[d.iloc[i]['Label']] = d.iloc[i]['keywords'].split(',')


data = training_data

# get a list of all categories to train for
categories = list(data.keys())
words = []
# a list of tuples with words in the sentence and category name
docs = []

for each_category in data.keys():
    for each_sentence in data[each_category]:
        # remove any punctuation from the sentence
        each_sentence = remove_punctuation(each_sentence)
        print(each_sentence)
        # extract words from each sentence and append to the word list
        w = nltk.word_tokenize(each_sentence)
        words.extend(w)
        docs.append((w, each_category))


# stem and lower each word and remove duplicates
stemmer = LancasterStemmer()
words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))

with open('Words_pid.txt', 'wb') as f:
    pickle.dump(words, f)


print(words)
print(docs)

# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(categories)

for doc in docs:
    if(doc[0] != []):
        # initialize our bag of words(bow) for each document in the list
        bow = []
        # list of tokenized words for the pattern
        token_words = doc[0]
        # stem each word
        token_words = [stemmer.stem(word.lower()) for word in token_words]
        # create our bag of words array
        for w in words:
            bow.append(1) if w in token_words else bow.append(0)
        output_row = list(output_empty)
        output_row[categories.index(doc[1])] = 1
        # our training set will contain a the bag of words model and the output row that tells
        # which catefory that bow belongs to.
        training.append([bow, output_row])

# shuffle our features and turn into np.array as tensorflow  takes in numpy array
random.shuffle(training)
training = np.array(training)

# trainX contains the Bag of words and train_y contains the label/ category
train_x = list(training[:,0])
train_y = list(training[:,1])


# reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=100, batch_size=8, show_metric=True)
model.save('fa_model_cdets.tflearn')


######Testing on 2018 data######
df = pd.read_csv('pid_sr_asr9k.csv', encoding = 'utf-8')
df = df.dropna(subset=['SR_Keywords'])
df = df.dropna(subset=['FA_CDETS'])
df = df[df['Year'] == 2018]

fa_label = df['FA_CDETS'].tolist()
keywords = df['SR_Keywords'].values.tolist()
for i in range(len(keywords)):
    keywords[i] = keywords[i].replace('[], ', '')
    keywords[i] = keywords[i].replace('], [', ',')

docs = []
for i in range(len(keywords)):
    s = ''
    if(len(keywords[i].replace(',', '').replace(' ', '')) > 2):
        for keyword in literal_eval(keywords[i])[:]:
            #print(keyword)
            keyword = keyword.split(':')[0]
            #s = s + (' '.join(set(keyword.split()))) + ','
            s = s + keyword + ','
            #print(s)
        docs.append(s.rstrip(','))
    else:
        docs.append(' ')

with open('Words_pid.txt', 'rb') as f:
    words = pickle.load(f)

new_docs = []
for each_sentence in docs:
    # remove any punctuation from the sentence
    each_sentence = remove_punctuation(each_sentence.replace(',', ', '))
    # extract words from each sentence and append to the word list
    w = nltk.word_tokenize(each_sentence)
    #print ("tokenized words: ", w)
    new_docs.append(' '.join(w))


tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model2 = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
#model.fit(train_x, train_y, n_epoch=100, batch_size=8, show_metric=True)
model2.load('fa_model_cdets.tflearn')

def get_tf_record(sentence):
    global words
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    # bag of words
    bow = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1
    return(np.array(bow))

categories_pred = []
for doc in new_docs:
    a = np.argsort(model2.predict([get_tf_record(doc)])[0])[-5:]
    pred = []
    for i in a:
        pred.append(categories[i])
    categories_pred.append(pred)

c = 0
for i in range(len(fa_label)):
    if(fa_label[i] in categories_pred[i]):
        c = c + 1

print('Accuracy: ', float(c/len(fa_label)))
df['CDETS_Prediction'] = categories_pred
df.to_csv('CDETS_Predictions_2018.csv', index = False, encoding = 'utf-8')

k = 0
for i in list(set(fa_label)):
    if(i in categories):
        k = k + 1

print(float(k/len(list(set(fa_label)))))
