######SR clustering on 2017 data######
import pandas as pd
from ast import literal_eval
import re
from collections import Counter
import math
import numpy as np
import pandas as pd
from gensim import corpora
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from ast import literal_eval
import gensim
from gensim.models import Phrases
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from pprint import pprint
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from nltk.corpus import stopwords 
import re

def unique_words(sentence, number):
    return [w for w in set(sentence.translate(None, punctuation).lower().split()) if len(w) >= number]

def preprocessing(line):
    line = line.lower()
    line = re.sub(r"[{}]".format(string.punctuation), " ", line)
    return line


df1 = pd.read_csv("IQS_All_SRNotes_Extracted.csv", encoding='utf-8')
df2 = pd.read_csv("IQS_FARC_Extracted_ASR9000.csv", encoding='utf-8')
df3 = pd.read_csv("SRNums_2017.csv", encoding='utf-8')
df4 = pd.read_csv("SRNums_2018.csv", encoding='utf-8')

sr_2017 = list(df3['SR_NUMBER'])
df1 = df1[df1['sr_number'].isin(sr_2017)]

d_2017 = df1.merge(df2, left_on='sr_number', right_on='SR_NUMBER', how='left')
#d['Keywords_x'] = d['Keywords_x'].apply(str)
d_2017 = d_2017.groupby('sr_number').agg({'sr_number':'first', 'Keywords_x': 'first', 'SYMPTOM_CODE': 'first'})
#d.to_csv('Joined_data_2017.csv', index = False, encoding='utf-8')
#d = d.drop(d.index[[3, 17, 26, 41, 75, 78, 80, 97, 105, 110, 148, 171, 258, 259, 262, 263, 282, 284, 299, 300, 316, 363, 402]])
d_2017 = d_2017.drop(d_2017.index[[0]])
##d = pd.read_csv("Joined_data_2017.csv", encoding='utf-8')
keywords = d_2017['Keywords_x'].values.tolist()
for i in range(len(keywords)):
    keywords[i] = keywords[i].replace('[], ', '')
    keywords[i] = keywords[i].replace('], [', ',')

docs_2017 = []
for i in range(len(keywords)):
    s = ''
    for keyword in literal_eval(keywords[i])[:]:
        #print(keyword)
        keyword = keyword.split(':')[0]
        keyword = ' '.join([i for i in keyword.split(' ') if not i.isdigit()])
        keyword = ' '.join([i for i in keyword.split(' ') if '/' not in i])
        keyword = re.sub("\S*\d\S*", "", keyword).strip()
        for w in keyword:
            if((ord(w) < 65 or ord(w) > 124) and (ord(w) is not 32) and (ord(w) is not 95)):
                keyword = keyword.replace(w, '')
        #s = s + (' '.join(set(keyword.split()))) + ','
        s = s + keyword + ','
    docs_2017.append(s.rstrip(','))

new_docs_2017 = []
for doc in docs_2017:
    # Remove Hex numbers followed by spaces
    doc = re.sub(r"(0x)+[0-9a-f]+\s*","",doc,0)
    # Remove Hex numbers followed by .
    doc = re.sub(r"(0x)+[0-9a-f]+\.","",doc,0)
    # Remove Binary numbers followed by spaces
    doc = re.sub(r"(0b)+[0-9a-f]+\s*","",doc,0)
    # Remove Binary numbers followed by .
    doc = re.sub(r"(0b)+[0-9a-f]+\.","",doc,0)
    #doc = re.sub(',', ' ', doc, 0)
    doc = doc.split(',')
    doc = ' '.join(list(set(doc))).strip()
    new_docs_2017.append(doc)

######2018 data######
######SR clustering on 2017 data######
df1_2018 = pd.read_csv("IQS_All_SRNotes_Extracted.csv", encoding='utf-8')
df2_2018 = pd.read_csv("IQS_FARC_Extracted_ASR9000.csv", encoding='utf-8')
df3 = pd.read_csv("SRNums_2017.csv", encoding='utf-8')
df4 = pd.read_csv("SRNums_2018.csv", encoding='utf-8')

sr_2018 = list(df4['SR_NUMBER'])
df1_2018 = df1_2018[df1_2018['sr_number'].isin(sr_2018)]

d_2018 = df1_2018.merge(df2_2018, left_on='sr_number', right_on='SR_NUMBER', how='left')

d_2018 = d_2018.groupby('sr_number').agg({'sr_number':'first', 'Keywords_x': ', '.join, 'SYMPTOM_CODE': 'first'})
#d.to_csv('Joined_data_2017.csv', index = False, encoding='utf-8')
#d_2018 = d_2018.drop(d_2018.index[[3, 17, 26, 41, 75, 78, 80, 97, 105, 110, 148, 171, 258, 259, 262, 263, 282, 284, 299, 300, 316, 363, 402]])
#d = pd.read_csv("Joined_data_2017.csv", encoding='utf-8')
keywords_2018 = d_2018['Keywords_x'].values.tolist()
for i in range(len(keywords_2018)):
    keywords_2018[i] = keywords_2018[i].replace('[], ', '')
    keywords_2018[i] = keywords_2018[i].replace('], [', ',')

docs_2018 = []
for i in range(len(keywords_2018)):
    s = ''
    for keyword in literal_eval(keywords_2018[i])[:]:
        #print(keyword)
        keyword = keyword.split(':')[0]
        keyword = ' '.join([i for i in keyword.split(' ') if not i.isdigit()])
        keyword = ' '.join([i for i in keyword.split(' ') if '/' not in i])
        keyword = re.sub("\S*\d\S*", "", keyword).strip()
        for w in keyword:
            if((ord(w) < 65 or ord(w) > 124) and (ord(w) is not 32) and (ord(w) is not 95)):
                keyword = keyword.replace(w, '')
        #s = s + (' '.join(set(keyword.split()))) + ','
        s = s + keyword + ','
    docs_2018.append(s.rstrip(','))

new_docs_2018 = []
for doc in docs_2018:
    # Remove Hex numbers followed by spaces
    doc = re.sub(r"(0x)+[0-9a-f]+\s*","",doc,0)
    # Remove Hex numbers followed by .
    doc = re.sub(r"(0x)+[0-9a-f]+\.","",doc,0)
    # Remove Binary numbers followed by spaces
    doc = re.sub(r"(0b)+[0-9a-f]+\s*","",doc,0)
    # Remove Binary numbers followed by .
    doc = re.sub(r"(0b)+[0-9a-f]+\.","",doc,0)
    #doc = re.sub(',', ' ', doc, 0)
    doc = doc.split(',')
    doc = ' '.join(list(set(doc))).strip()
    new_docs_2018.append(doc)




docs = docs_2018 + docs_2017
new_docs = new_docs_2018 + new_docs_2017
tokenizer = RegexpTokenizer(r'\w+')
for idx in range(len(docs)):
    docs[idx] = docs[idx].lower()  # Convert to lowercase.
    docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

# Remove numbers, but not words that contain numbers.
docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

# Remove words that are only one character.
docs = [[token for token in doc if len(token) > 1] for doc in docs]

# Lemmatize all words in documents.
lemmatizer = WordNetLemmatizer()
docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]


# Add bigrams and trigrams to docs (only ones that appear 5 times or more).
bigram = Phrases(docs, min_count=3)
bigram_mod = gensim.models.phrases.Phraser(bigram)
for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            if(token not in docs[idx]):
                docs[idx].append(token)

trigram = Phrases(bigram[docs], min_count = 3)#threshold=100
trigram_mod = gensim.models.phrases.Phraser(trigram)
for idx in range(len(docs)):
    for token in trigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            if(token not in docs[idx]):
                docs[idx].append(token)


print(trigram_mod[bigram_mod[docs[0]]])

# Remove rare and common tokens.
# Create a dictionary representation of the documents.
data_lemmatized = docs
dictionary = Dictionary(docs)

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=10, no_above=0.4)

stop = set(stopwords.words('english'))
stop_words = ['name', 'traceback', 'time', 'require', 'create', 'yamanashi', 'int', 'byte', 'lyt', 'still', 'thu', 'total', 'cisco', 'type', 'actual', 'node', 'show', 'needed', 'init', 'clear', 'set', 'ok', 'please', 'jan', 'feb', 'mar', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'mon', 'tue', 'wed', 'thur', 'fri', 'sat', 'sun', 'utc', 'ist', 'changed', 'info', 'saved', 'successfully', 'need', 'collecting', 'second', 'minute', 'hour', 'timer', 'timed', 'manager', 'director', 'major', 'fujitsu', 'us', 'india', 'united states', 'japan', 'china']
stop_words = list(set(list(stop) + stop_words))

for word in stop_words:
    if(word in dictionary.token2id):
        dictionary.filter_tokens(bad_ids=[dictionary.token2id[word]])

# Vectorize data
# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in docs]
print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))


# Set training parameters.
num_topics = 8
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, alpha='auto', eta='auto',iterations=iterations, num_topics=num_topics, passes=passes, eval_every=eval_every)#gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

# Can take a long time to run...
corpus_old = corpus.copy()
corpus = corpus[len(docs_2018):]
model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=data_lemmatized, start=2, limit=25, step=3)

# Print the coherence scores
limit=25; start=2; step=3;
x = range(start, limit, step)
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

# Show graph
limit=25; start=2; step=3;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


optimal_model = model_list[5]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))

x = optimal_model.show_topics(num_topics=100, num_words=50,formatted=False)
topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]

topic_keywords = []
#Below Code Prints Topics and Words
for topic,words in topics_words:
    topic_keywords.append(words)
    print(str(topic)+ "::"+ str(words))

#Pruning top keywords
for keywords in topic_keywords:
    for word in keywords:
        if word in stop_words:
            keywords.remove(word)


doc_topics =  []
prob = []
c=0
for doc in corpus_old:
    a = sorted(optimal_model[doc], key=lambda x: x[1])[-1]
    doc_topics.append(a[0])
    prob.append(a[1])
    c = c + 1

d_2017['cluster_id'] = doc_topics[len(docs_2018):]
d_2017['probability'] = prob[len(docs_2018):]

d_2018['cluster_id'] = doc_topics[:len(docs_2018)]
d_2018['probability'] = prob[:len(docs_2018)]

keywords = []
for cluster in doc_topics:
    keywords.append(topic_keywords[cluster])

d_2018['topic_keywords'] = keywords[:len(docs_2018)]
d_2017['topic_keywords'] = keywords[len(docs_2018):]

d_2018.to_csv('ASR9000_sr_keywords_clustered_2018_new.csv', encoding='utf-8')
d_2017.to_csv('ASR9000_sr_keywords_clustered_2017_new.csv', encoding='utf-8')














WORD = re.compile(r'\w+')
def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])
     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)
     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(str(text))
     return Counter(words)

c = 0
overall_similarity = []
overall_similarity_prob = []
for doc_2018 in new_docs_2018:
    print(c)
    c = c + 1
    doc_similarity = []
    vector1 = text_to_vector(doc_2018)
    for doc_2017 in new_docs_2017:
        vector2 = text_to_vector(doc_2017)
        doc_similarity.append(get_cosine(vector1, vector2))
    top_5_idx = np.argsort(doc_similarity)[-5:]
    top_5_values = [doc_similarity[i] for i in top_5_idx]
    similar_sr_nums = []
    for idx in top_5_idx:
        similar_sr_nums.append(sr_2017[idx])
    overall_similarity.append(similar_sr_nums)
    overall_similarity_prob.append(top_5_values)

d_2018['Similar_SR'] = overall_similarity
d_2018['Similar_SR_Probability'] = overall_similarity_prob

df = d_2018.merge(df2_2018, left_on='sr_number', right_on='SR_NUMBER', how='left')
df = df.groupby('sr_number').agg({'sr_number':'first', 'Keywords_x': 'first', 'SYMPTOM_CODE_x': 'first', 'cluster_id':'first', 'probability':'first', 'topic_keywords': 'first', 'Similar_SR': 'first', 'Similar_SR_Probability': 'first', 'FA_CASE_NUMBER': ', '.join})
df.to_csv('ASR9000_clustered_2018_similar_SR_new.csv', encoding='utf-8')


#########Creating new dataframe for collection#########
d = pd.DataFrame()
d['SR_NUMBER'] = df['sr_number'].astype(int)
d['SR_Keywords'] = df['Keywords_x']
d['Cluster_id'] = df['cluster_id'].astype(int)
d['Topic_Keywords'] = df['topic_keywords']
#d['Probability'] = df['probability']
d['Similar_SR'] = df['Similar_SR']
d['Similar_SR_Probability'] = df['Similar_SR_Probability']
d['FA_CASE_NUMBER'] = df['FA_CASE_NUMBER']
d['Year'] = [2018]*df.shape[0]

d1 = pd.DataFrame()
d1['SR_NUMBER'] = d_2017['sr_number'].astype(int)
d1['SR_Keywords'] = d_2017['Keywords_x']
d1['Cluster_id'] = d_2017['cluster_id'].astype(int)
d1['Topic_Keywords'] = d_2017['topic_keywords']
#d1['Probability'] = d_2017['probability']
d1['Year'] = [2017]*d_2017.shape[0]


d = d.append(d1)
d.to_csv('IQS_SR_Clusters_ASR9000.csv', encoding='utf-8', index = False)


c = 0
good_cases = []
for i in range(len(ass_fa)):
    for j in ass_fa[i]:
        if(j in rec_fa[i]):
            c = c + 1
        else:
            for k in rec_fa[i]:
                for l in d[j]:
                   if(l in d[k]):
                       c = c + 1
                       good_cases.append([j, k])
                       break
